#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
OpenLibrary dumps -> MySQL raw importer

- Creates DB/tables automatically
- Imports huge OL dumps in streaming mode
- Stores JSON without ASCII escaping (ensure_ascii=False)
- Resumable by byte-offset (import_state table)
"""

import argparse
import datetime as dt
import json
import os
import re
import sys
import time
from typing import Optional, Tuple

try:
    import pymysql
except ImportError:
    print("ERROR: pymysql is not installed. Install: pip install pymysql", file=sys.stderr)
    sys.exit(2)

# ----------------------------
# Helpers
# ----------------------------

SAFE_TABLE_RE = re.compile(r"[^a-z0-9_]+")


def safe_table_name(filename: str) -> str:
    """
    ol_dump_editions_latest.txt -> ol_dump_editions_latest
    then normalize to lowercase and [a-z0-9_]
    """
    base = os.path.basename(filename)
    name = os.path.splitext(base)[0].lower()
    name = SAFE_TABLE_RE.sub("_", name).strip("_")
    if not name:
        name = "ol_dump"
    return name


def parse_last_modified(value: str) -> Optional[str]:
    """
    OL often uses ISO timestamps; but formats may vary.
    We store as a normalized string (VARCHAR) to avoid failing on format changes.
    """
    v = (value or "").strip()
    if not v:
        return None
    # Keep as-is; optional normalization can be added later
    return v[:64]


def try_extract_from_json(obj: dict) -> Tuple[Optional[str], Optional[str]]:
    """
    Returns (ol_key, ol_type) if present.
    """
    ol_key = obj.get("key")
    ol_type = None

    t = obj.get("type")
    # type can be string "/type/edition" or dict {"key":"/type/edition"}
    if isinstance(t, str):
        ol_type = t
    elif isinstance(t, dict):
        ol_type = t.get("key")

    if isinstance(ol_key, str):
        ol_key = ol_key[:255]
    else:
        ol_key = None

    if isinstance(ol_type, str):
        ol_type = ol_type[:255]
    else:
        ol_type = None

    return ol_key, ol_type


def parse_dump_line(line: str) -> Tuple[Optional[str], Optional[int], Optional[str], Optional[str], Optional[str]]:
    """
    Tries to parse one OL dump line.
    Returns: (ol_key, revision, last_modified, ol_type, data_json)

    Strategy:
    - If line contains tabs, split into parts
    - Find the JSON part (first part that starts with "{" and ends with "}" after strip)
    - revision is first integer-looking field near start
    - last_modified is last non-json field often at end
    - If JSON parses, pull key/type from JSON when possible
    """
    s = line.strip("\n")
    if not s:
        return None, None, None, None, None

    parts = s.split("\t")
    json_part = None
    json_index = None

    # Identify JSON chunk
    for i, p in enumerate(parts):
        ps = p.strip()
        if ps.startswith("{") and ps.endswith("}"):
            json_part = ps
            json_index = i
            break

    # Some dumps can be pure JSON lines (no tabs)
    if json_part is None and s.lstrip().startswith("{") and s.rstrip().endswith("}"):
        json_part = s.strip()
        parts = [s.strip()]
        json_index = 0

    if json_part is None:
        return None, None, None, None, None

    # Try parse JSON
    try:
        obj = json.loads(json_part)
    except Exception:
        return None, None, None, None, None

    # Re-dump JSON without ASCII escaping
    data_json = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))

    # Try extract key/type from JSON
    ol_key, ol_type = try_extract_from_json(obj)

    # Guess ol_key from first part if JSON didn't have it
    if not ol_key and len(parts) > 0 and json_index is not None and json_index > 0:
        # Often first column is the key like "/works/OL123W"
        first = parts[0].strip()
        if first.startswith("/"):
            ol_key = first[:255]

    # Guess revision: first integer-like token in non-json parts
    revision = None
    for i, p in enumerate(parts):
        if i == json_index:
            continue
        ps = p.strip()
        if ps.isdigit():
            try:
                revision = int(ps)
                break
            except Exception:
                pass

    # Guess last_modified: last non-json field (often at end)
    last_modified = None
    if len(parts) >= 2:
        for i in range(len(parts) - 1, -1, -1):
            if i == json_index:
                continue
            cand = parts[i].strip()
            if cand and len(cand) <= 64:
                last_modified = parse_last_modified(cand)
                break

    return ol_key, revision, last_modified, ol_type, data_json


# ----------------------------
# MySQL
# ----------------------------

def mysql_connect(host: str, port: int, user: str, password: str, db: Optional[str], charset: str):
    return pymysql.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        database=db,
        charset=charset,
        autocommit=False,
        cursorclass=pymysql.cursors.DictCursor,
        write_timeout=600,
        read_timeout=600,
    )


def ensure_database(conn, db_name: str, charset: str, collation: str):
    with conn.cursor() as cur:
        cur.execute(
            f"CREATE DATABASE IF NOT EXISTS `{db_name}` "
            f"CHARACTER SET {charset} COLLATE {collation}"
        )
    conn.commit()


def ensure_meta_tables(conn, db_name: str, charset: str, collation: str):
    """
    import_state stores offsets for resumable import
    """
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


def ensure_dump_table(conn, db_name: str, table: str, charset: str, collation: str):
    sql = f"""
    CREATE TABLE IF NOT EXISTS `{db_name}`.`{table}` (
        `id` BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,

        -- для идемпотентности на уровне строки дампа
        `source_file` VARCHAR(512) NOT NULL,
        `source_offset` BIGINT UNSIGNED NOT NULL,

        `ol_key` VARCHAR(255) NULL,
        `revision` INT NULL,
        `last_modified` VARCHAR(64) NULL,
        `ol_type` VARCHAR(255) NULL,
        `data_json` LONGTEXT NOT NULL,
        `created_at` DATETIME NOT NULL,

        PRIMARY KEY (`id`),

        -- ключ записи (если есть)
        UNIQUE KEY `ux_ol_key` (`ol_key`),

        -- ключ строки дампа (всегда есть)
        UNIQUE KEY `ux_source_line` (`source_file`, `source_offset`),

        KEY `ix_type` (`ol_type`),
        KEY `ix_last_modified` (`last_modified`)
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
        if row and "byte_offset" in row:
            return int(row["byte_offset"] or 0)
    return 0


def save_state(conn, db_name: str, source_file: str, table: str, byte_offset: int):
    now = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    with conn.cursor() as cur:
        cur.execute(
            f"""
            INSERT INTO `{db_name}`.`import_state` (`source_file`, `table_name`, `byte_offset`, `updated_at`)
            VALUES (%s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
              `byte_offset`=VALUES(`byte_offset`),
              `updated_at`=VALUES(`updated_at`)
            """,
            (source_file, table, int(byte_offset), now),
        )
    conn.commit()


# ----------------------------
# Import logic
# ----------------------------

def import_file(
    conn,
    db_name: str,
    file_path: str,
    table: str,
    charset: str,
    collation: str,
    batch_size: int,
    commit_every: int,
    error_log_path: str,
):
    ensure_dump_table(conn, db_name, table, charset, collation)

    start_offset = load_state(conn, db_name, file_path, table)
    file_size = os.path.getsize(file_path)

    inserted = 0
    skipped = 0
    failed = 0

    t0 = time.time()
    last_commit_time = time.time()

    insert_sql = f"""
        INSERT INTO `{db_name}`.`{table}`
            (`ol_key`, `revision`, `last_modified`, `ol_type`, `data_json`, `created_at`)
        VALUES
            (%s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            `revision`=VALUES(`revision`),
            `last_modified`=VALUES(`last_modified`),
            `ol_type`=VALUES(`ol_type`),
            `data_json`=VALUES(`data_json`);
    """

    def log_error(msg: str):
        with open(error_log_path, "a", encoding="utf-8") as ef:
            ef.write(msg.rstrip("\n") + "\n")

    now_str = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        if start_offset > 0:
            # Seek by bytes: need binary mode. We'll reopen in binary for accurate seek.
            pass

    # Reopen in binary to seek accurately by byte offset, then decode line-by-line.
    with open(file_path, "rb") as fb:
        if start_offset > 0:
            fb.seek(start_offset)

        buf = []
        bytes_since_state = 0

        while True:
            pos_before = fb.tell()
            raw = fb.readline()
            if not raw:
                break

            pos_after = fb.tell()
            bytes_since_state += (pos_after - pos_before)

            try:
                line = raw.decode("utf-8", errors="replace")
            except Exception:
                failed += 1
                log_error(f"[DECODE_FAIL] file={file_path} pos={pos_before}")
                continue

            ol_key, revision, last_modified, ol_type, data_json = parse_dump_line(line)
            if not data_json:
                skipped += 1
                continue

            buf.append((ol_key, revision, last_modified, ol_type, data_json, now_str))

            if len(buf) >= batch_size:
                try:
                    with conn.cursor() as cur:
                        cur.executemany(insert_sql, buf)
                    inserted += len(buf)
                    buf.clear()
                except Exception as e:
                    failed += len(buf)
                    log_error(f"[BATCH_FAIL] file={file_path} pos={pos_before} err={repr(e)}")
                    buf.clear()
                    # continue, do not stop

            # Commit and save state periodically
            if inserted > 0 and (inserted % commit_every == 0):
                try:
                    conn.commit()
                except Exception as e:
                    log_error(f"[COMMIT_FAIL] file={file_path} pos={pos_before} err={repr(e)}")
                    # try continue

                # save byte offset for resumable import
                try:
                    save_state(conn, db_name, file_path, table, fb.tell())
                    bytes_since_state = 0
                except Exception as e:
                    log_error(f"[STATE_SAVE_FAIL] file={file_path} pos={pos_before} err={repr(e)}")

                # Progress log to stdout
                elapsed = max(0.001, time.time() - t0)
                mb_done = fb.tell() / (1024 * 1024)
                mb_total = file_size / (1024 * 1024)
                speed = mb_done / elapsed
                print(
                    f"[{table}] inserted={inserted} skipped={skipped} failed={failed} "
                    f"offset={fb.tell()} ({mb_done:.1f}/{mb_total:.1f} MiB) speed={speed:.2f} MiB/s"
                )

        # Flush remaining buffer
        if buf:
            try:
                with conn.cursor() as cur:
                    cur.executemany(insert_sql, buf)
                inserted += len(buf)
                buf.clear()
            except Exception as e:
                failed += len(buf)
                log_error(f"[FINAL_BATCH_FAIL] file={file_path} err={repr(e)}")
                buf.clear()

        # Final commit + state
        try:
            conn.commit()
        except Exception as e:
            log_error(f"[FINAL_COMMIT_FAIL] file={file_path} err={repr(e)}")

        try:
            save_state(conn, db_name, file_path, table, fb.tell())
        except Exception as e:
            log_error(f"[FINAL_STATE_SAVE_FAIL] file={file_path} err={repr(e)}")

    elapsed = max(0.001, time.time() - t0)
    print(f"[{table}] DONE inserted={inserted} skipped={skipped} failed={failed} time={elapsed:.1f}s")


def main():
    p = argparse.ArgumentParser(description="Import OpenLibrary dumps into MySQL (raw tables).")
    p.add_argument("--db", default=os.getenv("MYSQL_DB", "apidbooks"))
    p.add_argument("--host", default=os.getenv("MYSQL_HOST", "127.0.0.1"))
    p.add_argument("--port", type=int, default=int(os.getenv("MYSQL_PORT", "3306")))
    p.add_argument("--user", default=os.getenv("MYSQL_USER", "root"))
    p.add_argument("--password", default=os.getenv("MYSQL_PASSWORD", ""))
    p.add_argument("--charset", default=os.getenv("MYSQL_CHARSET", "utf8mb4"))
    p.add_argument("--collation", default=os.getenv("MYSQL_COLLATION", "utf8mb4_general_ci"))
    p.add_argument("--create-db", action="store_true", help="Create database if not exists.")
    p.add_argument("--dir", default="/home/apid/tmp", help="Directory with ol_dump_* files.")
    p.add_argument("--files", nargs="*", default=[], help="Explicit file paths. If empty, scans --dir for ol_dump_*.txt")
    p.add_argument("--batch-size", type=int, default=1000)
    p.add_argument("--commit-every", type=int, default=20000, help="Commit every N inserted rows.")
    p.add_argument("--error-log", default="ol_import_errors.log")
    args = p.parse_args()

    # Connect without db first if creating db
    conn0 = mysql_connect(args.host, args.port, args.user, args.password, None, args.charset)

    if args.create_db:
        ensure_database(conn0, args.db, args.charset, args.collation)

    # Connect to target db
    conn0.close()
    conn = mysql_connect(args.host, args.port, args.user, args.password, args.db, args.charset)

    ensure_meta_tables(conn, args.db, args.charset, args.collation)

    # collect files
    files = list(args.files)
    if not files:
        for name in sorted(os.listdir(args.dir)):
            if name.startswith("ol_dump_") and name.endswith(".txt"):
                files.append(os.path.join(args.dir, name))

    if not files:
        print("No dump files found. Use --files or --dir.", file=sys.stderr)
        sys.exit(1)

    # error log path absolute
    error_log_path = args.error_log
    if not os.path.isabs(error_log_path):
        error_log_path = os.path.join(os.getcwd(), error_log_path)

    print(f"DB={args.db} host={args.host}:{args.port} charset={args.charset} collation={args.collation}")
    print(f"Error log: {error_log_path}")
    print(f"Files: {len(files)}")

    for fp in files:
        table = safe_table_name(fp)
        print(f"\n=== Import: {fp} -> {args.db}.{table} ===")
        import_file(
            conn=conn,
            db_name=args.db,
            file_path=fp,
            table=table,
            charset=args.charset,
            collation=args.collation,
            batch_size=args.batch_size,
            commit_every=args.commit_every,
            error_log_path=error_log_path,
        )

    conn.close()
    print("\nALL DONE")


if __name__ == "__main__":
    main()
