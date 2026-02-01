#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import re
import json
import csv
import time
import datetime as dt
from typing import Optional, Tuple, List

try:
    import pymysql
except ImportError:
    print("ERROR: pymysql is not installed. Install inside venv: python -m pip install pymysql", file=sys.stderr)
    sys.exit(2)

SAFE_TABLE_RE = re.compile(r"[^a-z0-9_]+")
JSON_DECODER = json.JSONDecoder()


# ----------------- .env -----------------

def load_dotenv(path: str) -> dict:
    cfg = {}
    if not os.path.exists(path):
        return cfg
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip()
            if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
                v = v[1:-1]
            cfg[k] = v
    return cfg


def env_get(cfg: dict, key: str, default: str = "") -> str:
    return os.getenv(key, cfg.get(key, default))


# ----------------- helpers -----------------

def safe_table_name(filename: str) -> str:
    base = os.path.basename(filename)
    name = os.path.splitext(base)[0].lower()
    name = SAFE_TABLE_RE.sub("_", name).strip("_")
    return name or "ol_dump"


def json_compact_no_ascii(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def file_size_bytes(path: str) -> int:
    return os.path.getsize(path)


def read_first_line_bytes(path: str, max_bytes: int = 4096) -> bytes:
    with open(path, "rb") as f:
        return f.readline(max_bytes)


def count_tsv_columns_from_bytes(line: bytes) -> int:
    s = line.rstrip(b"\r\n")
    if not s:
        return 0
    return s.count(b"\t") + 1


def detect_kind_by_filename(path: str) -> str:
    """
    - tsv_covers_metadata: covers_metadata
    - tsv_wikidata: wikidata
    - tsv_generic: ratings / reading-log
    - json: остальные
    """
    base = os.path.basename(path).lower()
    if "covers_metadata" in base:
        return "tsv_covers_metadata"
    if "wikidata" in base:
        return "tsv_wikidata"
    if "ratings" in base or "reading-log" in base:
        return "tsv_generic"
    return "json"


def parse_json_line(line: str) -> Tuple[Optional[str], Optional[int], Optional[str], Optional[str]]:
    """
    Для json-дампов (works/editions/authors/..): ищем первый JSON объект/массив в строке.
    """
    s = line.rstrip("\n")
    if not s:
        return None, None, None, None

    i_obj = s.find("{")
    i_arr = s.find("[")
    if i_obj == -1 and i_arr == -1:
        return None, None, None, None

    if i_obj == -1:
        i0 = i_arr
    elif i_arr == -1:
        i0 = i_obj
    else:
        i0 = min(i_obj, i_arr)

    tail = s[i0:].lstrip()
    if not tail:
        return None, None, None, None

    try:
        obj, _end = JSON_DECODER.raw_decode(tail)
    except Exception:
        return None, None, None, None

    try:
        data_json = json_compact_no_ascii(obj)
    except Exception:
        return None, None, None, None

    ol_key = None
    if isinstance(obj, dict):
        k = obj.get("key")
        if isinstance(k, str) and k:
            ol_key = k[:255]

    parts = s.split("\t")
    revision = None
    for p in parts:
        ps = p.strip()
        if ps.isdigit():
            try:
                revision = int(ps)
                break
            except Exception:
                pass

    last_modified = None
    for p in reversed(parts):
        cand = p.strip()
        if cand and len(cand) <= 64:
            last_modified = cand[:64]
            break

    return ol_key, revision, last_modified, data_json


def null_marker_str() -> str:
    return chr(92) + "N"


def tsv_escape_basic(val) -> str:
    """
    Для обычных коротких колонок: экранируем backslash + табы/переносы.
    """
    nm = null_marker_str()
    if val is None:
        return nm
    s = str(val)
    s = s.replace("\\", "\\\\").replace("\t", "\\t").replace("\n", "\\n").replace("\r", "\\r")
    return s


def tsv_escape_json_text(val: str) -> str:
    r"""
    Для JSON-текста:
    - НЕ трогаем backslash (иначе сломаем \\uXXXX и прочую разметку)
    - убираем реальные табы/переносы, чтобы LOAD DATA не развалился
    """
    if val is None:
        return null_marker_str()
    s = str(val)
    # реальные \t/\n/\r заменяем на пробелы (или можно на \\n, но это будет уже изменение содержимого)
    s = s.replace("\t", " ").replace("\n", " ").replace("\r", " ")
    return s

def normalize_wikidata_json_text(s: str) -> str:
    """
    Wikidata dump часто хранит JSON как CSV-quoted строку:
      "{""id"": ""Q..."" , ...}"
    Это не JSON, пока не:
      - снять внешние кавычки
      - заменить "" -> "
    """
    if s is None:
        return ""

    t = str(s).strip()

    # Если JSON упакован в кавычки: " {...} "
    if len(t) >= 2 and t[0] == '"' and t[-1] == '"':
        t = t[1:-1]
        t = t.replace('""', '"')

    # Иногда кавычек снаружи нет, но "" внутри есть
    if '"":"' not in t and '""' in t and (t.startswith("{") or t.startswith("[")):
        # оставим как есть — это уже JSON-вид
        pass

    return t


def extract_json_from_line_wikidata(line: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Возвращает (wd_id, json_text) или (None, None)
    Поддерживает:
      - id<TAB>"{""id"":...}"
      - id<TAB>{...}
      - строка где JSON начинается не с 1-го символа (ищем первый { или [)
    """
    s = (line or "").rstrip("\n")
    if not s:
        return None, None

    # чаще всего: first_col \t json_blob
    if "\t" in s:
        first, rest = s.split("\t", 1)
        wd_id = first.strip()[:64] if first.strip() else None
        jt = normalize_wikidata_json_text(rest)
        if jt:
            # если json не в начале — попробуем сдвинуться к первому {/[.
            i_obj = jt.find("{")
            i_arr = jt.find("[")
            i0 = -1
            if i_obj != -1 and i_arr != -1:
                i0 = min(i_obj, i_arr)
            elif i_obj != -1:
                i0 = i_obj
            elif i_arr != -1:
                i0 = i_arr
            if i0 > 0:
                jt = jt[i0:].lstrip()

        return wd_id, jt

    # fallback: искать JSON прямо в строке
    i_obj = s.find("{")
    i_arr = s.find("[")
    if i_obj == -1 and i_arr == -1:
        return None, None
    i0 = i_obj if i_arr == -1 else (i_arr if i_obj == -1 else min(i_obj, i_arr))
    jt = normalize_wikidata_json_text(s[i0:].lstrip())
    return None, jt

# ----------------- mysql -----------------

def mysql_connect(host: str, port: int, user: str, password: str, db: Optional[str], charset: str, unix_socket: Optional[str]):
    kwargs = dict(
        user=user,
        password=password,
        database=db,
        charset=charset,
        autocommit=False,
        cursorclass=pymysql.cursors.DictCursor,
        write_timeout=600,
        read_timeout=600,
        local_infile=True,
    )
    if unix_socket:
        kwargs["unix_socket"] = unix_socket
    else:
        kwargs["host"] = host
        kwargs["port"] = port
    return pymysql.connect(**kwargs)


def ensure_database(conn, db_name: str, charset: str, collation: str):
    with conn.cursor() as cur:
        cur.execute(
            f"CREATE DATABASE IF NOT EXISTS `{db_name}` "
            f"CHARACTER SET {charset} COLLATE {collation}"
        )
    conn.commit()


def ensure_meta_tables(conn, db_name: str, charset: str, collation: str):
    sql = f"""
CREATE TABLE IF NOT EXISTS `{db_name}`.`import_state` (
  `id` BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  `source_file` VARCHAR(512) NOT NULL,
  `table_name` VARCHAR(128) NOT NULL,
  `byte_offset` BIGINT UNSIGNED NOT NULL DEFAULT 0,
  `updated_at` DATETIME NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `ux_file_table` (`source_file`, `table_name`)
) ENGINE=InnoDB
DEFAULT CHARSET={charset}
COLLATE={collation};
"""
    with conn.cursor() as cur:
        cur.execute(sql)
    conn.commit()


def load_state(conn, db_name: str, source_file: str, table: str) -> int:
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT `byte_offset` FROM `{db_name}`.`import_state` "
            f"WHERE `source_file`=%s AND `table_name`=%s LIMIT 1",
            (source_file, table),
        )
        row = cur.fetchone()
        if row and row.get("byte_offset") is not None:
            return int(row["byte_offset"] or 0)
    return 0


def save_state(conn, db_name: str, source_file: str, table: str, byte_offset: int):
    now = dt.datetime.now(dt.UTC).strftime("%Y-%m-%d %H:%M:%S")
    with conn.cursor() as cur:
        cur.execute(
            f"""
INSERT INTO `{db_name}`.`import_state` (`source_file`, `table_name`, `byte_offset`, `updated_at`)
VALUES (%s, %s, %s, %s)
ON DUPLICATE KEY UPDATE
  `byte_offset`=VALUES(`byte_offset`),
  `updated_at`=VALUES(`updated_at`);
""",
            (source_file, table, int(byte_offset), now),
        )
    conn.commit()


def is_file_done(conn, db_name: str, file_path: str, table: str) -> bool:
    size = file_size_bytes(file_path)
    off = load_state(conn, db_name, file_path, table)
    return off >= size


def table_has_columns(conn, db_name: str, table: str, required: List[str]) -> bool:
    req = set(required)
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT COLUMN_NAME
            FROM information_schema.COLUMNS
            WHERE TABLE_SCHEMA=%s AND TABLE_NAME=%s
            """,
            (db_name, table),
        )
        cols = {r["COLUMN_NAME"] for r in cur.fetchall()}
    return req.issubset(cols)


# ----------------- schema -----------------

def ensure_json_table(conn, db_name: str, table: str, charset: str, collation: str):
    sql = f"""
CREATE TABLE IF NOT EXISTS `{db_name}`.`{table}` (
  `id` BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,

  `source_file` VARCHAR(512) NOT NULL,
  `source_offset` BIGINT UNSIGNED NOT NULL,

  `ol_key` VARCHAR(255) NULL,
  `revision` INT NULL,
  `last_modified` VARCHAR(64) NULL,
  `data_json` LONGTEXT NOT NULL,
  `created_at` DATETIME NOT NULL,

  PRIMARY KEY (`id`),
  UNIQUE KEY `ux_source_line` (`source_file`, `source_offset`),
  UNIQUE KEY `ux_ol_key` (`ol_key`),
  KEY `ix_last_modified` (`last_modified`)
) ENGINE=InnoDB
DEFAULT CHARSET={charset}
COLLATE={collation};
"""
    with conn.cursor() as cur:
        cur.execute(sql)
    conn.commit()


def ensure_covers_metadata_table(conn, db_name: str, table: str, charset: str, collation: str):
    sql = f"""
CREATE TABLE IF NOT EXISTS `{db_name}`.`{table}` (
  `id` BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,

  `source_file` VARCHAR(512) NOT NULL,
  `source_offset` BIGINT UNSIGNED NOT NULL,

  `cover_id` BIGINT NULL,
  `width` INT NULL,
  `height` INT NULL,
  `created` DATE NULL,

  `created_at` DATETIME NOT NULL,

  PRIMARY KEY (`id`),
  UNIQUE KEY `ux_source_line` (`source_file`, `source_offset`),
  UNIQUE KEY `ux_cover_id` (`cover_id`)
) ENGINE=InnoDB
DEFAULT CHARSET={charset}
COLLATE={collation};
"""
    with conn.cursor() as cur:
        cur.execute(sql)
    conn.commit()


def ensure_generic_tsv_table(conn, db_name: str, table: str, ncols: int, charset: str, collation: str):
    """
    generic TSV: делаем LONGTEXT, чтобы не резать данные (особенно если попадётся большой field).
    """
    cols_sql = []
    for i in range(1, ncols + 1):
        cols_sql.append(f"`c{i}` LONGTEXT NULL")
    cols_block = ",\n  ".join(cols_sql)

    sql = f"""
CREATE TABLE IF NOT EXISTS `{db_name}`.`{table}` (
  `id` BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,

  `source_file` VARCHAR(512) NOT NULL,
  `source_offset` BIGINT UNSIGNED NOT NULL,

  {cols_block},

  `created_at` DATETIME NOT NULL,

  PRIMARY KEY (`id`),
  UNIQUE KEY `ux_source_line` (`source_file`, `source_offset`)
) ENGINE=InnoDB
DEFAULT CHARSET={charset}
COLLATE={collation};
"""
    with conn.cursor() as cur:
        cur.execute(sql)
    conn.commit()


def ensure_wikidata_table(conn, db_name: str, table: str, charset: str, collation: str):
    """
    Важно: если таблица уже была создана "не тем типом" (json-таблица),
    то пересоздаём ТОЛЬКО wikidata-таблицу автоматически.
    """
    required = ["source_file", "source_offset", "wd_id", "data_json", "created_at"]

    exists_ok = False
    try:
        exists_ok = table_has_columns(conn, db_name, table, required)
    except Exception:
        exists_ok = False

    if not exists_ok:
        # drop old/broken table
        with conn.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS `{db_name}`.`{table}`")
        conn.commit()

    sql = f"""
CREATE TABLE IF NOT EXISTS `{db_name}`.`{table}` (
  `id` BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,

  `source_file` VARCHAR(512) NOT NULL,
  `source_offset` BIGINT UNSIGNED NOT NULL,

  `wd_id` VARCHAR(64) NULL,
  `data_json` LONGTEXT NOT NULL,

  `created_at` DATETIME NOT NULL,

  PRIMARY KEY (`id`),
  UNIQUE KEY `ux_source_line` (`source_file`, `source_offset`),
  KEY `ix_wd_id` (`wd_id`)
) ENGINE=InnoDB
DEFAULT CHARSET={charset}
COLLATE={collation};
"""
    with conn.cursor() as cur:
        cur.execute(sql)
    conn.commit()


# ----------------- LOAD DATA builders -----------------

def build_load_json_sql(db_name: str, table: str) -> str:
    nm = null_marker_str()
    null_sql = "'" + nm + "'"

    sql = (
        "LOAD DATA LOCAL INFILE %s IGNORE\n"
        f"INTO TABLE `{db_name}`.`{table}`\n"
        "CHARACTER SET utf8mb4\n"
        "FIELDS TERMINATED BY '\\t'\n"
        "LINES TERMINATED BY '\\n'\n"
        "(@source_file, @source_offset, @ol_key, @revision, @last_modified, @data_json, @created_at)\n"
        "SET\n"
        f"  source_file = NULLIF(@source_file, {null_sql}),\n"
        "  source_offset = CAST(@source_offset AS UNSIGNED),\n"
        f"  ol_key = NULLIF(@ol_key, {null_sql}),\n"
        f"  revision = NULLIF(@revision, {null_sql}),\n"
        f"  last_modified = NULLIF(@last_modified, {null_sql}),\n"
        "  data_json = @data_json,\n"
        "  created_at = @created_at;\n"
    )
    return sql


def build_load_covers_sql(db_name: str, table: str) -> str:
    nm = null_marker_str()
    null_sql = "'" + nm + "'"

    sql = (
        "LOAD DATA LOCAL INFILE %s IGNORE\n"
        f"INTO TABLE `{db_name}`.`{table}`\n"
        "CHARACTER SET utf8mb4\n"
        "FIELDS TERMINATED BY '\\t'\n"
        "LINES TERMINATED BY '\\n'\n"
        "(@source_file, @source_offset, @cover_id, @width, @height, @created, @created_at)\n"
        "SET\n"
        f"  source_file = NULLIF(@source_file, {null_sql}),\n"
        "  source_offset = CAST(@source_offset AS UNSIGNED),\n"
        f"  cover_id = NULLIF(@cover_id, {null_sql}),\n"
        f"  width = NULLIF(@width, {null_sql}),\n"
        f"  height = NULLIF(@height, {null_sql}),\n"
        f"  created = NULLIF(@created, {null_sql}),\n"
        "  created_at = @created_at;\n"
    )
    return sql


def build_load_generic_tsv_sql(db_name: str, table: str, ncols: int) -> str:
    nm = null_marker_str()
    null_sql = "'" + nm + "'"

    vars_list = ["@source_file", "@source_offset"]
    for i in range(1, ncols + 1):
        vars_list.append(f"@c{i}")
    vars_list.append("@created_at")
    vars = ", ".join(vars_list)

    set_lines = [
        f"  source_file = NULLIF(@source_file, {null_sql})",
        "  source_offset = CAST(@source_offset AS UNSIGNED)",
    ]
    for i in range(1, ncols + 1):
        set_lines.append(f"  c{i} = NULLIF(@c{i}, {null_sql})")
    set_lines.append("  created_at = @created_at")

    set_sql = ",\n".join(set_lines)

    sql = (
        "LOAD DATA LOCAL INFILE %s IGNORE\n"
        f"INTO TABLE `{db_name}`.`{table}`\n"
        "CHARACTER SET utf8mb4\n"
        "FIELDS TERMINATED BY '\\t'\n"
        "LINES TERMINATED BY '\\n'\n"
        f"({vars})\n"
        "SET\n"
        f"{set_sql};\n"
    )
    return sql


def build_load_wikidata_sql(db_name: str, table: str) -> str:
    nm = null_marker_str()
    null_sql = "'" + nm + "'"

    sql = (
        "LOAD DATA LOCAL INFILE %s IGNORE\n"
        f"INTO TABLE `{db_name}`.`{table}`\n"
        "CHARACTER SET utf8mb4\n"
        "FIELDS TERMINATED BY '\\t'\n"
        "LINES TERMINATED BY '\\n'\n"
        "(@source_file, @source_offset, @wd_id, @data_json, @created_at)\n"
        "SET\n"
        f"  source_file = NULLIF(@source_file, {null_sql}),\n"
        "  source_offset = CAST(@source_offset AS UNSIGNED),\n"
        f"  wd_id = NULLIF(@wd_id, {null_sql}),\n"
        "  data_json = @data_json,\n"
        "  created_at = @created_at;\n"
    )
    return sql


# ----------------- import logic -----------------

def scan_dump_files(dump_dir: str) -> List[str]:
    return [
        os.path.join(dump_dir, f)
        for f in sorted(os.listdir(dump_dir))
        if f.startswith("ol_dump_") and f.endswith(".txt")
    ]


def import_json_file(conn, db_name: str, file_path: str, table: str, tmp_dir: str, chunk_lines: int, progress_every_chunks: int,
                     charset: str, collation: str):
    ensure_json_table(conn, db_name, table, charset, collation)

    if is_file_done(conn, db_name, file_path, table):
        print(f"[{table}] SKIP already done")
        return

    start_offset = load_state(conn, db_name, file_path, table)
    file_size = file_size_bytes(file_path)

    os.makedirs(tmp_dir, exist_ok=True)
    load_sql = build_load_json_sql(db_name, table)

    t0 = time.time()
    chunk_idx = 0
    total = 0
    written = 0
    skipped = 0
    failed = 0

    with open(file_path, "rb") as fb:
        if start_offset > 0:
            fb.seek(start_offset)

        while True:
            pos_chunk_start = fb.tell()
            tmp_path = os.path.join(tmp_dir, f"{table}.chunk_{chunk_idx}.tsv")
            lines_in_chunk = 0
            wrote_any = False
            now_str = dt.datetime.now(dt.UTC).strftime("%Y-%m-%d %H:%M:%S")

            with open(tmp_path, "w", encoding="utf-8", newline="\n") as out:
                while lines_in_chunk < chunk_lines:
                    pos_before = fb.tell()
                    raw = fb.readline()
                    if not raw:
                        break
                    total += 1
                    line = raw.decode("utf-8", errors="replace")

                    ol_key, revision, last_modified, data_json = parse_json_line(line)
                    if not data_json:
                        skipped += 1
                        continue

                    row = "\t".join([
                        tsv_escape_basic(file_path),
                        str(pos_before),
                        tsv_escape_basic(ol_key),
                        tsv_escape_basic(revision),
                        tsv_escape_basic(last_modified),
                        tsv_escape_json_text(data_json),
                        now_str,
                    ]) + "\n"
                    out.write(row)
                    wrote_any = True
                    written += 1
                    lines_in_chunk += 1

            if not wrote_any:
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

                if fb.tell() == pos_chunk_start:
                    save_state(conn, db_name, file_path, table, fb.tell())
                    break

                chunk_idx += 1
                continue

            try:
                with conn.cursor() as cur:
                    cur.execute(load_sql, (tmp_path,))
                conn.commit()
            except Exception as e:
                conn.rollback()
                raise RuntimeError(f"LOAD DATA failed for {tmp_path}: {repr(e)}")

            save_state(conn, db_name, file_path, table, fb.tell())

            try:
                os.remove(tmp_path)
            except Exception:
                pass

            chunk_idx += 1
            if chunk_idx % progress_every_chunks == 0:
                elapsed = max(0.001, time.time() - t0)
                mb_done = fb.tell() / (1024 * 1024)
                mb_total = file_size / (1024 * 1024)
                speed = mb_done / elapsed
                print(
                    f"[{table}] chunks={chunk_idx} total_lines={total} written={written} skipped={skipped} failed={failed} "
                    f"offset={fb.tell()} ({mb_done:.1f}/{mb_total:.1f} MiB) speed={speed:.2f} MiB/s"
                )

    elapsed = max(0.001, time.time() - t0)
    print(f"[{table}] DONE written={written} skipped={skipped} failed={failed} time={elapsed:.1f}s")


def import_covers_metadata(conn, db_name: str, file_path: str, table: str, tmp_dir: str, chunk_lines: int, progress_every_chunks: int,
                          charset: str, collation: str):
    ensure_covers_metadata_table(conn, db_name, table, charset, collation)

    if is_file_done(conn, db_name, file_path, table):
        print(f"[{table}] SKIP already done")
        return

    start_offset = load_state(conn, db_name, file_path, table)
    file_size = file_size_bytes(file_path)

    os.makedirs(tmp_dir, exist_ok=True)
    load_sql = build_load_covers_sql(db_name, table)

    t0 = time.time()
    chunk_idx = 0
    total = 0
    written = 0
    skipped = 0

    with open(file_path, "rb") as fb:
        if start_offset > 0:
            fb.seek(start_offset)

        while True:
            pos_chunk_start = fb.tell()
            tmp_path = os.path.join(tmp_dir, f"{table}.chunk_{chunk_idx}.tsv")
            lines_in_chunk = 0
            wrote_any = False
            now_str = dt.datetime.now(dt.UTC).strftime("%Y-%m-%d %H:%M:%S")

            with open(tmp_path, "w", encoding="utf-8", newline="\n") as out:
                while lines_in_chunk < chunk_lines:
                    pos_before = fb.tell()
                    raw = fb.readline()
                    if not raw:
                        break
                    total += 1

                    line = raw.decode("utf-8", errors="replace").rstrip("\n")
                    if not line:
                        skipped += 1
                        continue

                    cols = line.split("\t")
                    if len(cols) != 4:
                        skipped += 1
                        continue

                    cover_id, width, height, created = cols

                    row = "\t".join([
                        tsv_escape_basic(file_path),
                        str(pos_before),
                        tsv_escape_basic(cover_id),
                        tsv_escape_basic(width),
                        tsv_escape_basic(height),
                        tsv_escape_basic(created),
                        now_str,
                    ]) + "\n"
                    out.write(row)
                    wrote_any = True
                    written += 1
                    lines_in_chunk += 1

            if not wrote_any:
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

                if fb.tell() == pos_chunk_start:
                    save_state(conn, db_name, file_path, table, fb.tell())
                    break

                chunk_idx += 1
                continue

            try:
                with conn.cursor() as cur:
                    cur.execute(load_sql, (tmp_path,))
                conn.commit()
            except Exception as e:
                conn.rollback()
                raise RuntimeError(f"LOAD DATA failed for {tmp_path}: {repr(e)}")

            save_state(conn, db_name, file_path, table, fb.tell())

            try:
                os.remove(tmp_path)
            except Exception:
                pass

            chunk_idx += 1
            if chunk_idx % progress_every_chunks == 0:
                elapsed = max(0.001, time.time() - t0)
                mb_done = fb.tell() / (1024 * 1024)
                mb_total = file_size / (1024 * 1024)
                speed = mb_done / elapsed
                print(
                    f"[{table}] chunks={chunk_idx} total_lines={total} written={written} skipped={skipped} "
                    f"offset={fb.tell()} ({mb_done:.1f}/{mb_total:.1f} MiB) speed={speed:.2f} MiB/s"
                )

    elapsed = max(0.001, time.time() - t0)
    print(f"[{table}] DONE written={written} skipped={skipped} time={elapsed:.1f}s")


def import_generic_tsv(conn, db_name: str, file_path: str, table: str, tmp_dir: str, chunk_lines: int, progress_every_chunks: int,
                       charset: str, collation: str):
    first = read_first_line_bytes(file_path)
    ncols = count_tsv_columns_from_bytes(first)
    if ncols <= 0:
        raise RuntimeError(f"TSV detect failed (empty first line): {file_path}")
    if ncols > 200:
        raise RuntimeError(f"TSV too many columns ({ncols}) in {file_path} (suspicious)")

    ensure_generic_tsv_table(conn, db_name, table, ncols, charset, collation)

    if is_file_done(conn, db_name, file_path, table):
        print(f"[{table}] SKIP already done")
        return

    start_offset = load_state(conn, db_name, file_path, table)
    file_size = file_size_bytes(file_path)

    os.makedirs(tmp_dir, exist_ok=True)
    load_sql = build_load_generic_tsv_sql(db_name, table, ncols)

    t0 = time.time()
    chunk_idx = 0
    total = 0
    written = 0
    skipped = 0

    # ВАЖНО: читаем как текст и используем csv.reader — он разворачивает "" -> "
    with open(file_path, "r", encoding="utf-8", newline="") as f:
        # для offset-ресюма нам нужен байтовый оффсет, поэтому отдельно открываем rb
        fb = open(file_path, "rb")
        try:
            if start_offset > 0:
                fb.seek(start_offset)
                f.seek(start_offset)  # в utf-8 это может быть неточно, но ratings/reading-log обычно ASCII; иначе fallback ниже

            reader = csv.reader(f, delimiter="\t", quotechar='"', doublequote=True)

            while True:
                tmp_path = os.path.join(tmp_dir, f"{table}.chunk_{chunk_idx}.tsv")
                lines_in_chunk = 0
                wrote_any = False
                now_str = dt.datetime.now(dt.UTC).strftime("%Y-%m-%d %H:%M:%S")
                pos_chunk_start = fb.tell()

                with open(tmp_path, "w", encoding="utf-8", newline="\n") as out:
                    while lines_in_chunk < chunk_lines:
                        pos_before = fb.tell()
                        raw = fb.readline()
                        if not raw:
                            break
                        total += 1

                        # распарсим эту же строку csv.reader-ом:
                        try:
                            line = raw.decode("utf-8", errors="replace")
                            row = next(csv.reader([line], delimiter="\t", quotechar='"', doublequote=True))
                        except Exception:
                            skipped += 1
                            continue

                        if len(row) != ncols:
                            skipped += 1
                            continue

                        row_fields = [tsv_escape_basic(file_path), str(pos_before)]
                        row_fields.extend(tsv_escape_json_text(c) for c in row)  # generic: не ломаем backslash
                        row_fields.append(now_str)
                        out.write("\t".join(row_fields) + "\n")
                        wrote_any = True
                        written += 1
                        lines_in_chunk += 1

                if not wrote_any:
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass

                    if fb.tell() == pos_chunk_start:
                        save_state(conn, db_name, file_path, table, fb.tell())
                        break

                    chunk_idx += 1
                    continue

                try:
                    with conn.cursor() as cur:
                        cur.execute(load_sql, (tmp_path,))
                    conn.commit()
                except Exception as e:
                    conn.rollback()
                    raise RuntimeError(f"LOAD DATA failed for {tmp_path}: {repr(e)}")

                save_state(conn, db_name, file_path, table, fb.tell())

                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

                chunk_idx += 1
                if chunk_idx % progress_every_chunks == 0:
                    elapsed = max(0.001, time.time() - t0)
                    mb_done = fb.tell() / (1024 * 1024)
                    mb_total = file_size / (1024 * 1024)
                    speed = mb_done / elapsed
                    print(
                        f"[{table}] chunks={chunk_idx} total_lines={total} written={written} skipped={skipped} "
                        f"offset={fb.tell()} ({mb_done:.1f}/{mb_total:.1f} MiB) speed={speed:.2f} MiB/s"
                    )

        finally:
            try:
                fb.close()
            except Exception:
                pass

    elapsed = max(0.001, time.time() - t0)
    print(f"[{table}] DONE written={written} skipped={skipped} time={elapsed:.1f}s")


def import_wikidata(conn, db_name: str, file_path: str, table: str, tmp_dir: str, chunk_lines: int, progress_every_chunks: int,
                    charset: str, collation: str):
    """
    Wikidata формат: <QID>\t"<JSON with ""doublequotes"">\n
    Поэтому читаем строго csv.reader(delimiter=\t, quotechar=").
    """
    ensure_wikidata_table(conn, db_name, table, charset, collation)

    if is_file_done(conn, db_name, file_path, table):
        print(f"[{table}] SKIP already done")
        return

    start_offset = load_state(conn, db_name, file_path, table)
    file_size = file_size_bytes(file_path)

    os.makedirs(tmp_dir, exist_ok=True)
    load_sql = build_load_wikidata_sql(db_name, table)

    t0 = time.time()
    chunk_idx = 0
    total = 0
    written = 0
    skipped = 0
    bad_json = 0

    with open(file_path, "rb") as fb:
        if start_offset > 0:
            fb.seek(start_offset)

        while True:
            pos_chunk_start = fb.tell()
            tmp_path = os.path.join(tmp_dir, f"{table}.chunk_{chunk_idx}.tsv")
            lines_in_chunk = 0
            wrote_any = False
            now_str = dt.datetime.now(dt.UTC).strftime("%Y-%m-%d %H:%M:%S")

            with open(tmp_path, "w", encoding="utf-8", newline="\n") as out:
                while lines_in_chunk < chunk_lines:
                    pos_before = fb.tell()
                    raw = fb.readline()
                    if not raw:
                        break

                    total += 1
                    line = raw.decode("utf-8", errors="replace").rstrip("\n")

                    # ✅ ключевой момент: csv.reader сам снимет внешние кавычки и превратит "" в "
                    try:
                        cols = next(csv.reader([line], delimiter="\t", quotechar='"', doublequote=True))
                    except Exception:
                        skipped += 1
                        continue

                    if not cols or len(cols) < 2:
                        skipped += 1
                        continue

                    wd_id = cols[0].strip()[:64] if cols[0] else None
                    json_text = cols[1].strip() if cols[1] else ""

                    if not json_text:
                        skipped += 1
                        continue

                    jt = json_text.lstrip()
                    if not (jt.startswith("{") or jt.startswith("[")):
                        skipped += 1
                        continue

                    # ✅ проверяем, что это реально JSON
                    try:
                        obj, _end = JSON_DECODER.raw_decode(jt)
                    except Exception:
                        bad_json += 1
                        skipped += 1
                        continue

                    # берём id из JSON если есть
                    if isinstance(obj, dict):
                        jid = obj.get("id")
                        if isinstance(jid, str) and jid:
                            wd_id = jid[:64]

                    data_json = json_compact_no_ascii(obj)

                    row = "\t".join([
                        tsv_escape_basic(file_path),
                        str(pos_before),
                        tsv_escape_basic(wd_id),
                        tsv_escape_json_text(data_json),
                        now_str,
                    ]) + "\n"
                    out.write(row)
                    wrote_any = True
                    written += 1
                    lines_in_chunk += 1

            if not wrote_any:
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

                if fb.tell() == pos_chunk_start:
                    save_state(conn, db_name, file_path, table, fb.tell())
                    break

                chunk_idx += 1
                continue

            try:
                with conn.cursor() as cur:
                    cur.execute(load_sql, (tmp_path,))
                conn.commit()
            except Exception as e:
                conn.rollback()
                raise RuntimeError(f"LOAD DATA failed for {tmp_path}: {repr(e)}")

            save_state(conn, db_name, file_path, table, fb.tell())

            try:
                os.remove(tmp_path)
            except Exception:
                pass

            chunk_idx += 1
            if chunk_idx % progress_every_chunks == 0:
                elapsed = max(0.001, time.time() - t0)
                mb_done = fb.tell() / (1024 * 1024)
                mb_total = file_size / (1024 * 1024)
                speed = mb_done / elapsed
                print(
                    f"[{table}] chunks={chunk_idx} total_lines={total} written={written} skipped={skipped} bad_json={bad_json} "
                    f"offset={fb.tell()} ({mb_done:.1f}/{mb_total:.1f} MiB) speed={speed:.2f} MiB/s"
                )

    elapsed = max(0.001, time.time() - t0)
    print(f"[{table}] DONE written={written} skipped={skipped} bad_json={bad_json} time={elapsed:.1f}s")

# ----------------- main -----------------

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dotenv_path = os.path.join(script_dir, ".env")
    cfg = load_dotenv(dotenv_path)

    mysql_db = env_get(cfg, "MYSQL_DB", "apidbooks")
    mysql_user = env_get(cfg, "MYSQL_USER", "root")
    mysql_password = env_get(cfg, "MYSQL_PASSWORD", "")
    mysql_host = env_get(cfg, "MYSQL_HOST", "127.0.0.1")
    mysql_port = int(env_get(cfg, "MYSQL_PORT", "3306") or "3306")
    mysql_unix_socket = env_get(cfg, "MYSQL_UNIX_SOCKET", "").strip() or None
    mysql_charset = env_get(cfg, "MYSQL_CHARSET", "utf8mb4")
    mysql_collation = env_get(cfg, "MYSQL_COLLATION", "utf8mb4_general_ci")

    dump_dir = env_get(cfg, "DUMPS_DIR", "/home/apid/tmp")
    tmp_dir = env_get(cfg, "TMP_TSV_DIR", "/home/apid/tmp/ol_tmp_tsv")
    chunk_lines = int(env_get(cfg, "CHUNK_LINES", "300000") or "300000")
    progress_every = int(env_get(cfg, "PROGRESS_EVERY_CHUNKS", "1") or "1")
    create_db = env_get(cfg, "CREATE_DB", "1").strip().lower() in ("1", "true", "yes", "on")

    if not os.path.isdir(dump_dir):
        print(f"ERROR: DUMPS_DIR not found: {dump_dir}", file=sys.stderr)
        sys.exit(1)

    conn0 = mysql_connect(mysql_host, mysql_port, mysql_user, mysql_password, None, mysql_charset, mysql_unix_socket)
    if create_db:
        ensure_database(conn0, mysql_db, mysql_charset, mysql_collation)
    conn0.close()

    conn = mysql_connect(mysql_host, mysql_port, mysql_user, mysql_password, mysql_db, mysql_charset, mysql_unix_socket)

    try:
        with conn.cursor() as cur:
            cur.execute("SHOW VARIABLES LIKE 'local_infile'")
            row = cur.fetchone()
            if row and str(row.get("Value", "")).upper() == "OFF":
                print("ERROR: MySQL server local_infile=OFF. Enable: SET GLOBAL local_infile=1;", file=sys.stderr)
                conn.close()
                sys.exit(2)
    except Exception:
        print("WARN: Could not verify local_infile variable", file=sys.stderr)

    ensure_meta_tables(conn, mysql_db, mysql_charset, mysql_collation)

    files = [
        os.path.join(dump_dir, f)
        for f in sorted(os.listdir(dump_dir))
        if f.startswith("ol_dump_") and f.endswith(".txt")
    ]
    if not files:
        print(f"ERROR: No ol_dump_*.txt files found in {dump_dir}", file=sys.stderr)
        conn.close()
        sys.exit(1)

    print(f"DB={mysql_db} user={mysql_user} host={mysql_host}:{mysql_port} socket={mysql_unix_socket or '-'}")
    print(f"DUMPS_DIR={dump_dir}")
    print(f"TMP_TSV_DIR={tmp_dir}")
    print(f"CHUNK_LINES={chunk_lines}")
    print(f"FILES={len(files)}")

    for fp in files:
        table = safe_table_name(fp)
        kind = detect_kind_by_filename(fp)
        print(f"\n=== {fp} -> {mysql_db}.{table} ({kind}) ===")

        if kind == "tsv_covers_metadata":
            import_covers_metadata(conn, mysql_db, fp, table, tmp_dir, chunk_lines, progress_every, mysql_charset, mysql_collation)
        elif kind == "tsv_wikidata":
            # ВАЖНО: если раньше state поставил "done" при written=0 — надо сбросить state вручную (см. ниже).
            import_wikidata(conn, mysql_db, fp, table, tmp_dir, chunk_lines, progress_every, mysql_charset, mysql_collation)
        elif kind == "tsv_generic":
            import_generic_tsv(conn, mysql_db, fp, table, tmp_dir, chunk_lines, progress_every, mysql_charset, mysql_collation)
        else:
            import_json_file(conn, mysql_db, fp, table, tmp_dir, chunk_lines, progress_every, mysql_charset, mysql_collation)

    conn.close()
    print("\nALL DONE")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"FATAL: {repr(e)}", file=sys.stderr)
        sys.exit(1)
