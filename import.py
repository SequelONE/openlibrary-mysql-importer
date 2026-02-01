#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import sys
import json
import time
import argparse
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from collections import OrderedDict

import pymysql
from dotenv import load_dotenv

load_dotenv()


def env(name: str, default=None, cast=None):
    v = os.getenv(name, default)
    if v is None:
        return None
    if cast is not None:
        try:
            return cast(v)
        except Exception:
            raise RuntimeError(f"Invalid env {name}={v!r}")
    return v


# ==================== PROGRESS (NEW) ====================

class Progress:
    """
    ✅ Динамичный прогресс:
    - сразу показывает 0%
    - умеет "пульсировать" (spinner) во время долгих операций
    """
    def __init__(self, kind: str, total: int, started_at: float, bar_len: int = 24):
        self.kind = kind
        self.total = max(0, int(total))
        self.started_at = started_at
        self.bar_len = bar_len
        self.spinner = "|/-\\"
        self.spin_i = 0
        self.last_line_len = 0
        self.last_print_at = 0.0
        self.last_processed = 0

    def _fmt_seconds(self, secs: float) -> str:
        if secs < 0 or secs == float("inf"):
            return "?"
        secs = int(secs)
        h = secs // 3600
        m = (secs % 3600) // 60
        s = secs % 60
        if h > 0:
            return f"{h}h{m:02d}m{s:02d}s"
        if m > 0:
            return f"{m}m{s:02d}s"
        return f"{s}s"

    def _render(self, processed: int, last_id: int, status: str = "") -> str:
        processed = int(processed)
        now = time.time()
        elapsed = now - self.started_at
        speed = processed / elapsed if elapsed > 0 else 0.0
        pct = (processed / self.total * 100.0) if self.total > 0 else 0.0
        remaining = (self.total - processed) if self.total > 0 else 0
        eta = (remaining / speed) if speed > 0 and self.total > 0 else float("inf")

        filled = int(self.bar_len * (pct / 100.0)) if self.total > 0 else 0
        filled = max(0, min(self.bar_len, filled))
        bar = "#" * filled + "-" * (self.bar_len - filled)

        if status:
            status = f" | {status}"

        return (
            f"{self.kind} [{bar}] {processed:,}/{self.total:,} ({pct:5.1f}%) "
            f"speed={speed:,.1f} rows/s ETA={self._fmt_seconds(eta)} last_id={last_id}{status}"
        )

    def print(self, processed: int, last_id: int, status: str = "", force_newline: bool = False):
        line = self._render(processed, last_id, status=status)
        pad = max(0, self.last_line_len - len(line))
        sys.stdout.write("\r" + line + (" " * pad))
        sys.stdout.flush()
        self.last_line_len = len(line)
        if force_newline:
            sys.stdout.write("\n")
            sys.stdout.flush()

    def pulse(self, processed: int, last_id: int, status: str):
        ch = self.spinner[self.spin_i % len(self.spinner)]
        self.spin_i += 1
        self.print(processed, last_id, status=f"{ch} {status}")

    def maybe_print(self, processed: int, last_id: int, every_rows: int, every_secs: int):
        now = time.time()
        if processed - self.last_processed >= every_rows or now - self.last_print_at >= every_secs:
            self.print(processed, last_id)
            self.last_print_at = now
            self.last_processed = processed


# -------------------- TABLE NAMES --------------------

@dataclass(frozen=True)
class TableNames:
    prefix: str
    ol_covers: str = "ol_dump_covers_metadata_latest"

    @property
    def books(self) -> str:
        return f"{self.prefix}books"

    @property
    def books_trans(self) -> str:
        return f"{self.prefix}books_trans"

    @property
    def editions(self) -> str:
        return f"{self.prefix}books_editions"

    @property
    def editions_trans(self) -> str:
        return f"{self.prefix}books_editions_trans"

    @property
    def authors(self) -> str:
        return f"{self.prefix}books_authors"

    @property
    def authors_trans(self) -> str:
        return f"{self.prefix}books_authors_trans"

    @property
    def covers(self) -> str:
        return f"{self.prefix}books_covers"

    @property
    def tags_books(self) -> str:
        return f"{self.prefix}books_tags"

    @property
    def pivot_books_tags(self) -> str:
        return f"{self.prefix}books_book_tag"

    @property
    def tags_editions(self) -> str:
        return f"{self.prefix}books_editions_tags"

    @property
    def pivot_editions_tags(self) -> str:
        return f"{self.prefix}books_editions_edition_tag"

    @property
    def tags_authors(self) -> str:
        return f"{self.prefix}books_authors_tags"

    @property
    def pivot_authors_tags(self) -> str:
        return f"{self.prefix}books_authors_author_tag"

    @property
    def source_records(self) -> str:
        return f"{self.prefix}books_source_records"

    @property
    def source_record_links(self) -> str:
        return f"{self.prefix}books_source_record_links"


# -------------------- PARSERS --------------------

RE_OL_NUM = re.compile(r"OL(\d+)[A-Z]$", re.IGNORECASE)
RE_YEAR = re.compile(r"\b(1[0-9]{3}|20[0-9]{2}|21[0-9]{2})\b")  # 1000-2199


def parse_ol_num(ol_key: str) -> Optional[int]:
    if not ol_key:
        return None
    tail = str(ol_key).strip().split("/")[-1]
    m = RE_OL_NUM.search(tail)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def json_load_safe(s: Any) -> Optional[dict]:
    if s is None:
        return None
    if isinstance(s, dict):
        return s
    try:
        return json.loads(s)
    except Exception:
        return None


def extract_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        t = value.strip()
        return t or None
    if isinstance(value, dict):
        v = value.get("value")
        if isinstance(v, str):
            t = v.strip()
            return t or None
    return None


def parse_year_to_date(value: Any) -> Optional[str]:
    """
    Требование: если попалось "год" или текст+год -> YYYY-01-01
    """
    if value is None:
        return None
    s = value.strip() if isinstance(value, str) else str(value).strip()
    if not s:
        return None
    if re.match(r"^\d{4}-\d{2}-\d{2}$", s):
        return s
    m = RE_YEAR.search(s)
    if not m:
        return None
    return f"{m.group(1)}-01-01"


def parse_ol_datetime(dj: dict, key: str) -> Optional[str]:
    """
    OL datetime: {"type":"/type/datetime","value":"2021-12-26T21:32:02.453026"}
    -> "2021-12-26 21:32:02"
    """
    v = dj.get(key)
    if not isinstance(v, dict):
        return None
    s = v.get("value")
    if not isinstance(s, str) or not s.strip():
        return None
    s = s.strip()
    s = s.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
    except Exception:
        try:
            base = s.split(".")[0]
            dt = datetime.fromisoformat(base)
        except Exception:
            return None
    return dt.strftime("%Y-%m-%d %H:%M:%S")


# -------------------- SLUG (strict latin) --------------------

CYR_MAP = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e",
    "ж": "zh", "з": "z", "и": "i", "й": "y", "к": "k", "л": "l", "м": "m",
    "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u",
    "ф": "f", "х": "h", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "sch",
    "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu", "я": "ya",
    "і": "i", "ї": "yi", "є": "ie", "ґ": "g",
}

try:
    from unidecode import unidecode  # optional
except Exception:
    unidecode = None


def to_ascii_latin(s: str) -> str:
    """
    Важно: это ТОЛЬКО для slug.
    title/name/base.name НЕ трогаем: Unicode должен сохраниться как в OL (öäü и т.д.)
    """
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)

    lowered = s.lower()
    tmp = []
    for ch in lowered:
        tmp.append(CYR_MAP.get(ch, ch))
    s = "".join(tmp)

    if unidecode is not None:
        s = unidecode(s)

    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    return s.lower()


def slugify_latin(s: str) -> Optional[str]:
    s = to_ascii_latin(s).strip()
    if not s:
        return None
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s or None


# -------------------- OL extractors --------------------

def author_full_name(dj: dict) -> Optional[str]:
    personal = extract_text(dj.get("personal_name")) or (dj.get("personal_name") if isinstance(dj.get("personal_name"), str) else None)
    last = extract_text(dj.get("last_name")) or (dj.get("last_name") if isinstance(dj.get("last_name"), str) else None)
    name = extract_text(dj.get("name")) or (dj.get("name") if isinstance(dj.get("name"), str) else None)

    parts = []
    if personal:
        parts.append(personal.strip())
    if last:
        if personal and last.strip().lower() in personal.lower():
            pass
        else:
            parts.append(last.strip())

    full = " ".join([p for p in parts if p]).strip()
    if full:
        return full
    if name:
        return name.strip() or None
    if personal:
        return personal.strip() or None
    return None


def extract_subjects(dj: dict) -> List[str]:
    keys = ["subjects", "subject_places", "subject_people", "subject_times", "genres"]
    raw: List[str] = []
    for k in keys:
        v = dj.get(k)
        if isinstance(v, list):
            for item in v:
                if isinstance(item, str):
                    t = item.strip()
                    if t:
                        raw.append(t)

    seen = set()
    out = []
    for s in raw:
        k = s.casefold()
        if k in seen:
            continue
        seen.add(k)
        out.append(s)
    return out


def extract_cover_ids(dj: dict, kind: str) -> List[int]:
    key = "covers" if kind in ("work", "edition") else "photos"
    v = dj.get(key)
    ids: List[int] = []
    if isinstance(v, list):
        for x in v:
            if isinstance(x, int):
                ids.append(x)
            elif isinstance(x, str) and x.isdigit():
                ids.append(int(x))

    seen = set()
    out = []
    for i in ids:
        if i in seen:
            continue
        seen.add(i)
        out.append(i)
    return out


def iter_source_records(dj: dict) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    sr = dj.get("source_records")
    if not isinstance(sr, list):
        return out
    for item in sr:
        if not isinstance(item, str):
            continue
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            source, key = item.split(":", 1)
        else:
            source, key = "unknown", item
        source = (source.strip() or "unknown")[:64]
        key = (key.strip())[:255]
        if not key:
            continue
        out.append((source, key))
    return out


# -------------------- CHECKPOINT --------------------

def checkpoint_dir() -> str:
    d = env("TMP_TSV_DIR", "/tmp")
    path = os.path.join(d, "books_etl_checkpoint")
    os.makedirs(path, exist_ok=True)
    return path


def checkpoint_path(source_table: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_]+", "_", source_table)
    return os.path.join(checkpoint_dir(), f"{safe}.json")


def load_checkpoint(source_table: str) -> Dict[str, Any]:
    p = checkpoint_path(source_table)
    if not os.path.exists(p):
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}


def save_checkpoint(source_table: str, data: Dict[str, Any]):
    p = checkpoint_path(source_table)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, p)

def load_simple_checkpoint(key: str) -> int:
    p = checkpoint_path(key)
    if not os.path.exists(p):
        return 0
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f) or {}
        return int(data.get("last_id", 0))
    except Exception:
        return 0


def save_simple_checkpoint(key: str, last_id: int):
    save_checkpoint(key, {"last_id": int(last_id), "updated_at": int(time.time())})


# -------------------- DB --------------------

def db_connect() -> pymysql.connections.Connection:
    # ✅ таймауты, чтобы не было "вечного зависания"
    connect_timeout = env("MYSQL_CONNECT_TIMEOUT", 10, int) or 10
    read_timeout = env("MYSQL_READ_TIMEOUT", 300, int) or 300
    write_timeout = env("MYSQL_WRITE_TIMEOUT", 300, int) or 300

    conn = pymysql.connect(
        host=env("MYSQL_HOST", "127.0.0.1"),
        port=env("MYSQL_PORT", 3306, int),
        user=env("MYSQL_USER"),
        password=env("MYSQL_PASSWORD"),
        database=env("MYSQL_DB"),
        charset=env("MYSQL_CHARSET", "utf8mb4"),
        autocommit=False,
        cursorclass=pymysql.cursors.DictCursor,
        connect_timeout=connect_timeout,
        read_timeout=read_timeout,
        write_timeout=write_timeout,
    )

    # ✅ ускорители (можно отключить через ENV)
    if env("BOOKS_FAST_SESSION", 1, int):
        try:
            with conn.cursor() as cur:
                cur.execute("SET SESSION foreign_key_checks=0")
                cur.execute("SET SESSION unique_checks=0")
        except Exception:
            pass

    return conn


def table_exists(conn, table: str) -> bool:
    sql = """
    SELECT 1 FROM information_schema.tables
    WHERE table_schema = DATABASE() AND table_name = %s
    LIMIT 1
    """
    with conn.cursor() as cur:
        cur.execute(sql, [table])
        return cur.fetchone() is not None


def has_columns(conn, table: str, columns: List[str]) -> bool:
    sql = """
    SELECT COUNT(*) AS c
    FROM information_schema.columns
    WHERE table_schema = DATABASE()
      AND table_name = %s
      AND column_name IN ({})
    """.format(",".join(["%s"] * len(columns)))
    with conn.cursor() as cur:
        cur.execute(sql, [table] + columns)
        row = cur.fetchone()
        return bool(row) and int(row["c"]) == len(columns)


def detect_table_prefix(conn) -> str:
    forced = env("MYSQL_TABLE_PREFIX", "")
    if forced is not None and forced != "":
        return forced
    if table_exists(conn, "sone_books"):
        return "sone_"
    return ""


def list_ol_json_tables(conn) -> List[str]:
    sql = """
    SELECT table_name AS table_name
    FROM information_schema.tables
    WHERE table_schema = DATABASE()
      AND table_name LIKE 'ol_dump\\_%'
    ORDER BY table_name
    """
    with conn.cursor() as cur:
        cur.execute(sql)
        rows = cur.fetchall()
    candidates = [(r.get("table_name") or r.get("TABLE_NAME")) for r in rows if isinstance(r, dict)]
    candidates = [t for t in candidates if t]
    tables = [t for t in candidates if has_columns(conn, t, ["ol_key", "data_json"])]
    return tables


def peek_type_stats(conn, table: str, sample: int = 1000) -> Dict[str, int]:
    sql = f"SELECT data_json FROM `{table}` LIMIT %s"
    stats: Dict[str, int] = {}
    with conn.cursor() as cur:
        cur.execute(sql, [sample])
        for r in cur.fetchall():
            dj = json_load_safe(r.get("data_json"))
            if not dj:
                continue
            t = dj.get("type")
            key = t.get("key") if isinstance(t, dict) else None
            if isinstance(key, str):
                stats[key] = stats.get(key, 0) + 1
    return stats


def classify_table(stats: Dict[str, int]) -> Optional[str]:
    candidates = {
        "work": stats.get("/type/work", 0),
        "edition": stats.get("/type/edition", 0),
        "author": stats.get("/type/author", 0),
    }
    best_kind, best_cnt = max(candidates.items(), key=lambda kv: kv[1])
    return best_kind if best_cnt > 0 else None


def detect_ol_sources(conn) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {"work": [], "edition": [], "author": []}
    for t in list_ol_json_tables(conn):
        kind = classify_table(peek_type_stats(conn, t, sample=1000))
        if kind:
            out[kind].append(t)
    return out


def source_has_id(conn, source_table: str) -> bool:
    return has_columns(conn, source_table, ["id"])


def get_max_id(conn, table: str) -> int:
    with conn.cursor() as cur:
        cur.execute(f"SELECT MAX(id) AS m FROM `{table}`")
        row = cur.fetchone()
        m = row.get("m") if isinstance(row, dict) else None
        return int(m or 0)


# -------------------- LRU CACHE --------------------

class LruDict:
    def __init__(self, max_size: int = 200_000):
        self.max_size = max_size
        self.d = OrderedDict()

    def get(self, k):
        if k in self.d:
            self.d.move_to_end(k)
            return self.d[k]
        return None

    def set(self, k, v):
        if k in self.d:
            self.d.move_to_end(k)
        self.d[k] = v
        if len(self.d) > self.max_size:
            self.d.popitem(last=False)


def chunked(items: List[Any], n: int):
    for i in range(0, len(items), n):
        yield items[i:i+n]


# -------------------- UPSERT HELPERS --------------------

def ensure_source_record(conn, source_records_table: str, source: str, source_key: str, created_at: Optional[str], updated_at: Optional[str]) -> int:
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT id FROM `{source_records_table}` WHERE source=%s AND source_key=%s LIMIT 1",
            [source, source_key],
        )
        row = cur.fetchone()
        if row:
            return int(row["id"])

        ca = created_at or time.strftime("%Y-%m-%d %H:%M:%S")
        ua = updated_at or ca

        cur.execute(
            f"INSERT INTO `{source_records_table}` (source, source_key, payload, created_at, updated_at) "
            f"VALUES (%s,%s,NULL,%s,%s)",
            [source, source_key, ca, ua],
        )
        return int(cur.lastrowid)


def link_source_record(conn, links_table: str, source_record_id: int, entity_type: str, entity_id: int, created_at: Optional[str], updated_at: Optional[str]):
    ca = created_at or time.strftime("%Y-%m-%d %H:%M:%S")
    ua = updated_at or ca
    with conn.cursor() as cur:
        cur.execute(
            f"INSERT IGNORE INTO `{links_table}` (source_record_id, entity_type, entity_id, created_at, updated_at) "
            f"VALUES (%s,%s,%s,%s,%s)",
            [source_record_id, entity_type, entity_id, ca, ua],
        )


def slug_candidates(base_slug: str, ol_num: int):
    yield base_slug
    yield f"{base_slug}-{ol_num}"
    n = 2
    while True:
        yield f"{base_slug}-{ol_num}-{n}"
        n += 1


def _dup_key_name_from_msg(msg: str) -> str:
    # pymysql/MySQL обычно пишет: Duplicate entry '...' for key 'some_key'
    m = re.search(r"for key '([^']+)'", msg)
    return m.group(1) if m else ""


def _is_dup_1062(e: Exception) -> bool:
    try:
        return getattr(e, "args", None) and int(e.args[0]) == 1062
    except Exception:
        return False


def insert_translation(
    batch_rows: List[Tuple[Any, ...]],
    fk_field: str,
    entity_id: int,
    locale: str,
    title_or_name: Optional[str],
    slug_source: Optional[str],
    content: Optional[str],
    ol_num: int,
    created_at: Optional[str],
    updated_at: Optional[str],
):
    """
    Теперь НЕ пишет в БД. Только добавляет кортеж значений в batch_rows.
    Реальная вставка будет executemany() в process_ol_table().
    """
    intro = content[:500] if content else None
    base_slug = slugify_latin(slug_source or title_or_name or "") or str(ol_num)

    # slug должен влезать в varchar(255)
    # оставляем запас под суффиксы
    if base_slug and len(base_slug) > 220:
        base_slug = base_slug[:220].rstrip("-")

    ca = created_at or time.strftime("%Y-%m-%d %H:%M:%S")
    ua = updated_at or ca

    # slug с суффиксом -olnum-<n> генерим сразу, но в БД вставим через INSERT IGNORE
    # чтобы не зависать на дубликатах слуга
    slug = f"{base_slug}-{ol_num}"

    batch_rows.append((
        entity_id, locale, title_or_name, slug,
        None, None, intro, content, ol_num, ca, ua
    ))


def insert_base_work_or_edition(
    conn,
    base_table: str,
    kind: str,
    site_id: int,
    category_id: int,
    ol_num: int,
    original_name: Optional[str],
    genres_raw: List[str],
    screenshots: List[int],
    created_at: Optional[str],
    updated_at: Optional[str],
) -> int:
    with conn.cursor() as cur:
        cur.execute(f"SELECT id, name FROM `{base_table}` WHERE ol_key=%s LIMIT 1", [ol_num])
        row = cur.fetchone()
        if row:
            if has_columns(conn, base_table, ["name"]) and (row.get("name") in (None, "")) and original_name:
                cur.execute(
                    f"UPDATE `{base_table}` SET name=%s, updated_at=%s WHERE id=%s",
                    [original_name, updated_at or time.strftime("%Y-%m-%d %H:%M:%S"), row["id"]],
                )
            return int(row["id"])

        cols = ["site_id", "category_id", "ol_key", "genres_raw", "screenshots", "status", "created_at", "updated_at"]
        vals: List[Any] = [
            site_id,
            category_id,
            ol_num,
            json.dumps(genres_raw, ensure_ascii=False) if genres_raw else None,
            json.dumps(screenshots, ensure_ascii=False) if screenshots else None,
            1,
            created_at or time.strftime("%Y-%m-%d %H:%M:%S"),
            updated_at or (created_at or time.strftime("%Y-%m-%d %H:%M:%S")),
        ]

        if has_columns(conn, base_table, ["name"]):
            cols.insert(2, "name")
            vals.insert(2, original_name)

        if kind == "edition":
            cols.insert(2, "book_id")
            vals.insert(2, None)

        sql = f"INSERT INTO `{base_table}` ({','.join(cols)}) VALUES ({','.join(['%s']*len(vals))})"
        cur.execute(sql, vals)
        return int(cur.lastrowid)


def insert_base_author(
    conn,
    authors_table: str,
    site_id: int,
    category_id: int,
    ol_num: int,
    original_name: Optional[str],
    genres_raw: List[str],
    screenshots: List[int],
    birth_date: Optional[str],
    death_date: Optional[str],
    created_at: Optional[str],
    updated_at: Optional[str],
) -> int:
    with conn.cursor() as cur:
        cur.execute(f"SELECT id, name FROM `{authors_table}` WHERE ol_key=%s LIMIT 1", [ol_num])
        row = cur.fetchone()
        if row:
            if has_columns(conn, authors_table, ["name"]) and (row.get("name") in (None, "")) and original_name:
                cur.execute(
                    f"UPDATE `{authors_table}` SET name=%s, updated_at=%s WHERE id=%s",
                    [original_name, updated_at or time.strftime("%Y-%m-%d %H:%M:%S"), row["id"]],
                )
            return int(row["id"])

        cols = ["site_id", "category_id", "ol_key", "genres_raw", "screenshots", "status", "created_at", "updated_at"]
        vals: List[Any] = [
            site_id,
            category_id,
            ol_num,
            json.dumps(genres_raw, ensure_ascii=False) if genres_raw else None,
            json.dumps(screenshots, ensure_ascii=False) if screenshots else None,
            1,
            created_at or time.strftime("%Y-%m-%d %H:%M:%S"),
            updated_at or (created_at or time.strftime("%Y-%m-%d %H:%M:%S")),
        ]

        if has_columns(conn, authors_table, ["name"]):
            cols.insert(2, "name")
            vals.insert(2, original_name)

        if has_columns(conn, authors_table, ["birth_date"]):
            cols.insert(2, "birth_date")
            vals.insert(2, birth_date)

        if has_columns(conn, authors_table, ["death_date"]):
            insert_pos = 2 + (1 if "birth_date" in cols else 0)
            cols.insert(insert_pos, "death_date")
            vals.insert(insert_pos, death_date)

        sql = f"INSERT INTO `{authors_table}` ({','.join(cols)}) VALUES ({','.join(['%s']*len(vals))})"
        cur.execute(sql, vals)
        return int(cur.lastrowid)


# -------------------- TAGS (BULK) --------------------

def bulk_fetch_tag_ids(conn, tags_table: str, slugs: List[str], in_limit: int = 800) -> Dict[str, int]:
    if not slugs:
        return {}
    out: Dict[str, int] = {}
    with conn.cursor() as cur:
        for part in chunked(slugs, in_limit):
            placeholders = ",".join(["%s"] * len(part))
            cur.execute(
                f"SELECT id, slug FROM `{tags_table}` WHERE slug IN ({placeholders})",
                part
            )
            for row in cur.fetchall():
                s = row.get("slug")
                if s:
                    out[s] = int(row["id"])
    return out


def bulk_insert_missing_tags(conn, tags_table: str, slug_to_name: Dict[str, str], created_at: Optional[str], updated_at: Optional[str]):
    if not slug_to_name:
        return
    ca = created_at or time.strftime("%Y-%m-%d %H:%M:%S")
    ua = updated_at or ca
    rows = [(slug_to_name[slug], slug, ca, ua) for slug in slug_to_name.keys()]
    with conn.cursor() as cur:
        cur.executemany(
            f"INSERT IGNORE INTO `{tags_table}` (name, slug, items_count, created_at, updated_at) "
            f"VALUES (%s,%s,0,%s,%s)",
            rows
        )


def bulk_insert_pivots(conn, pivot_table: str, left_col: str, pairs: List[Tuple[int, int]], created_at: Optional[str], updated_at: Optional[str]):
    if not pairs:
        return
    ca = created_at or time.strftime("%Y-%m-%d %H:%M:%S")
    ua = updated_at or ca
    rows = [(lid, tid, ca, ua) for (lid, tid) in pairs]
    with conn.cursor() as cur:
        cur.executemany(
            f"INSERT IGNORE INTO `{pivot_table}` ({left_col}, tag_id, created_at, updated_at) "
            f"VALUES (%s,%s,%s,%s)",
            rows
        )


# -------------------- COVERS --------------------

def import_covers_metadata(conn, tn: TableNames):
    if not table_exists(conn, tn.covers):
        raise RuntimeError(f"Target table '{tn.covers}' not found.")
    if not table_exists(conn, tn.ol_covers):
        print(f"SKIP covers: {tn.ol_covers} not found")
        return

    # ✅ настройки чанков
    chunk = env("COVERS_CHUNK", 50000, int) or 50000
    checkpoint_key = f"covers_{tn.ol_covers}"

    # ✅ определяем max(cover_id) для прогресса (если cover_id есть)
    max_id = 0
    with conn.cursor() as cur:
        try:
            cur.execute(f"SELECT MAX(cover_id) AS m FROM `{tn.ol_covers}`")
            row = cur.fetchone()
            max_id = int((row or {}).get("m") or 0)
        except Exception:
            max_id = 0

    last_id = load_simple_checkpoint(checkpoint_key)
    total = max(0, max_id - last_id) if max_id > 0 else 1

    print(f"START covers: {tn.ol_covers} -> {tn.covers} (resume last_cover_id={last_id}, max_cover_id={max_id})")

    t0 = time.time()
    progress = Progress(kind="covers", total=total, started_at=t0)
    progress.print(0, last_id, status="start")

    inserted_total = 0

    while True:
        # ✅ пульс перед запросом — видно, что жив
        progress.pulse(inserted_total if max_id > 0 else 0, last_id, f"chunk after_cover_id={last_id} limit={chunk}")

        # ⚠️ ВАЖНО: берём только cover_id > last_id, сортируем, ограничиваем LIMIT
        # Это делает запрос коротким и предсказуемым.
        sql = (
            f"INSERT IGNORE INTO `{tn.covers}` (cover_id, width, height, created_at, updated_at) "
            f"SELECT cover_id, width, height, NOW(), NOW() "
            f"FROM `{tn.ol_covers}` "
            f"WHERE cover_id > %s "
            f"ORDER BY cover_id "
            f"LIMIT %s"
        )

        with conn.cursor() as cur:
            cur.execute(sql, [last_id, chunk])
            inserted = cur.rowcount

            # ✅ узнаём новый last_id (макс cover_id в выбранном чанке)
            cur.execute(
                f"SELECT MAX(cover_id) AS m FROM `{tn.ol_covers}` WHERE cover_id > %s ORDER BY cover_id LIMIT %s",
                [last_id, chunk],
            )
            # Этот запрос неидеален, поэтому делаем надёжнее так:
            cur.execute(
                f"SELECT cover_id FROM `{tn.ol_covers}` WHERE cover_id > %s ORDER BY cover_id LIMIT %s",
                [last_id, chunk],
            )
            rows = cur.fetchall()

        if not rows:
            conn.commit()
            break

        # rows — список dict'ов {'cover_id': ...}
        new_last = int(rows[-1]["cover_id"])
        last_id = new_last

        conn.commit()
        save_simple_checkpoint(checkpoint_key, last_id)

        inserted_total += max(0, int(inserted))

        # ✅ обновляем прогресс
        done = max(0, last_id - load_simple_checkpoint(checkpoint_key)) if max_id > 0 else inserted_total
        done = min(done, total) if total > 0 else done
        progress.print(done, last_id, status=f"inserted_total={inserted_total:,}")

        # ✅ если дошли до max_id — выходим
        if max_id > 0 and last_id >= max_id:
            break

    progress.print(total if total > 0 else inserted_total, last_id, status=f"done inserted_total={inserted_total:,}", force_newline=True)
    print(f"OK covers imported -> {tn.covers} (inserted_total={inserted_total:,})")


# -------------------- TAG COUNTS --------------------

def recalc_tag_counts(conn, tags_table: str, pivot_table: str):
    if not table_exists(conn, tags_table) or not table_exists(conn, pivot_table):
        print(f"SKIP recalc counts: missing {tags_table} or {pivot_table}")
        return
    with conn.cursor() as cur:
        cur.execute(f"UPDATE `{tags_table}` SET items_count=0")
        cur.execute(
            f"UPDATE `{tags_table}` t "
            f"JOIN (SELECT tag_id, COUNT(*) c FROM `{pivot_table}` WHERE deleted_at IS NULL GROUP BY tag_id) x "
            f"ON x.tag_id=t.id "
            f"SET t.items_count = x.c"
        )
    conn.commit()


# -------------------- PROCESSING --------------------

def process_ol_table(
    conn,
    tn: TableNames,
    source_table: str,
    kind: str,
    locale: str,
    site_id: int,
    category_id: int,
    batch: int,
    do_tags: bool,
    do_sources: bool,
):
    if kind == "work":
        base_table = tn.books
        trans_table = tn.books_trans
        fk_field = "book_id"
        title_field = "title"
        tags_table = tn.tags_books
        pivot_table = tn.pivot_books_tags
        pivot_left = "book_id"
        entity_type = "book"
    elif kind == "edition":
        base_table = tn.editions
        trans_table = tn.editions_trans
        fk_field = "edition_id"
        title_field = "title"
        tags_table = tn.tags_editions
        pivot_table = tn.pivot_editions_tags
        pivot_left = "edition_id"
        entity_type = "edition"
    elif kind == "author":
        base_table = tn.authors
        trans_table = tn.authors_trans
        fk_field = "author_id"
        title_field = "name"
        tags_table = tn.tags_authors
        pivot_table = tn.pivot_authors_tags
        pivot_left = "author_id"
        entity_type = "author"
    else:
        raise ValueError(kind)

    for required in [base_table, trans_table]:
        if not table_exists(conn, required):
            raise RuntimeError(f"Target table '{required}' not found.")

    if not source_has_id(conn, source_table):
        raise RuntimeError(f"Source table '{source_table}' has no id column -> resume/чанки нормально не сделать.")

    cp = load_checkpoint(source_table)
    last_id = int(cp.get("last_id", 0))

    max_id = get_max_id(conn, source_table)
    total = max(0, max_id - last_id)

    # частота вывода прогресса
    progress_every_rows = env("PROGRESS_EVERY_ROWS", 50000, int) or 50000
    progress_every_secs = env("PROGRESS_EVERY_SECS", 10, int) or 10

    # batch вставка translations
    trans_batch_size = env("TRANS_BATCH", 2000, int) or 2000  # поднял дефолт
    skip_trans = bool(env("BOOKS_SKIP_TRANSLATIONS", 0, int))

    # ⚠️ ВАЖНО: INSERT IGNORE = не висим на дублях slug
    trans_sql = (
        f"INSERT IGNORE INTO `{trans_table}` "
        f"({fk_field}, locale, {title_field}, slug, meta_title, meta_description, introtext, content, ol_key, created_at, updated_at) "
        f"VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
    )

    # ускорение тегов
    tag_cache = LruDict(max_size=env("BOOKS_TAG_CACHE_SIZE", 200000, int) or 200000)
    tag_in_limit = env("BOOKS_TAG_IN_LIMIT", 800, int) or 800

    print(f"START {kind}: {source_table} -> {base_table} (resume last_id={last_id}, max_id={max_id}, total={total})")

    t0 = time.time()
    progress = Progress(kind=kind, total=(total if total > 0 else 1), started_at=t0)
    progress.print(0, last_id, status="start")

    processed = 0
    commits = 0

    try:
        while True:
            # чтобы было видно, что жив даже если SELECT медленный
            progress.pulse(processed, last_id, f"select batch={batch}")

            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT id, ol_key, data_json FROM `{source_table}` "
                    f"WHERE id > %s ORDER BY id LIMIT %s",
                    [last_id, batch]
                )
                rows = cur.fetchall()

            if not rows:
                break

            max_seen = last_id

            # аккумуляторы tags
            batch_slug_to_name: Dict[str, str] = {}
            batch_entity_slug_pairs: List[Tuple[int, str]] = []
            batch_tags_ca: Optional[str] = None
            batch_tags_ua: Optional[str] = None

            # аккумулятор translations (executemany)
            trans_rows: List[Tuple[Any, ...]] = []

            for r in rows:
                src_id = int(r["id"])
                if src_id > max_seen:
                    max_seen = src_id

                processed += 1

                ol_num = parse_ol_num(r["ol_key"])
                if not ol_num:
                    continue

                dj = json_load_safe(r["data_json"])
                if not dj:
                    continue

                created_at = parse_ol_datetime(dj, "created")
                updated_at = parse_ol_datetime(dj, "last_modified") or created_at

                subjects = extract_subjects(dj)
                screenshots = extract_cover_ids(dj, kind)

                if kind == "work":
                    title = extract_text(dj.get("title"))
                    content = extract_text(dj.get("description"))

                    entity_id = insert_base_work_or_edition(
                        conn, base_table, "work",
                        site_id=site_id, category_id=category_id,
                        ol_num=ol_num,
                        original_name=title,
                        genres_raw=subjects,
                        screenshots=screenshots,
                        created_at=created_at,
                        updated_at=updated_at,
                    )

                    if not skip_trans:
                        insert_translation(
                            trans_rows, fk_field,
                            entity_id=entity_id,
                            locale=locale,
                            title_or_name=title,
                            slug_source=title,
                            content=content,
                            ol_num=ol_num,
                            created_at=created_at,
                            updated_at=updated_at,
                        )

                elif kind == "edition":
                    title = extract_text(dj.get("title"))
                    content = extract_text(dj.get("description"))

                    entity_id = insert_base_work_or_edition(
                        conn, base_table, "edition",
                        site_id=site_id, category_id=category_id,
                        ol_num=ol_num,
                        original_name=title,
                        genres_raw=subjects,
                        screenshots=screenshots,
                        created_at=created_at,
                        updated_at=updated_at,
                    )

                    if not skip_trans:
                        insert_translation(
                            trans_rows, fk_field,
                            entity_id=entity_id,
                            locale=locale,
                            title_or_name=title,
                            slug_source=title,
                            content=content,
                            ol_num=ol_num,
                            created_at=created_at,
                            updated_at=updated_at,
                        )

                elif kind == "author":
                    full_name = author_full_name(dj)
                    bio = extract_text(dj.get("bio")) or extract_text(dj.get("description"))
                    birth = parse_year_to_date(dj.get("birth_date"))
                    death = parse_year_to_date(dj.get("death_date"))

                    entity_id = insert_base_author(
                        conn, base_table,
                        site_id=site_id, category_id=category_id,
                        ol_num=ol_num,
                        original_name=full_name,
                        genres_raw=subjects,
                        screenshots=screenshots,
                        birth_date=birth,
                        death_date=death,
                        created_at=created_at,
                        updated_at=updated_at,
                    )

                    if not skip_trans:
                        insert_translation(
                            trans_rows, fk_field,
                            entity_id=entity_id,
                            locale=locale,
                            title_or_name=full_name,
                            slug_source=full_name,
                            content=bio,
                            ol_num=ol_num,
                            created_at=created_at,
                            updated_at=updated_at,
                        )

                # TAGS
                if do_tags and subjects:
                    if not table_exists(conn, tags_table) or not table_exists(conn, pivot_table):
                        raise RuntimeError(f"Tags tables missing: {tags_table} or {pivot_table}")

                    if batch_tags_ca is None:
                        batch_tags_ca = created_at
                    if batch_tags_ua is None:
                        batch_tags_ua = updated_at

                    for subj in subjects:
                        tag_slug = slugify_latin(subj)
                        if not tag_slug:
                            continue
                        if tag_slug not in batch_slug_to_name:
                            batch_slug_to_name[tag_slug] = subj
                        batch_entity_slug_pairs.append((entity_id, tag_slug))

                # SOURCES
                if do_sources:
                    if not table_exists(conn, tn.source_records) or not table_exists(conn, tn.source_record_links):
                        raise RuntimeError(f"Source tables missing: {tn.source_records} or {tn.source_record_links}")
                    for source, source_key in iter_source_records(dj):
                        sr_id = ensure_source_record(conn, tn.source_records, source, source_key, created_at, updated_at)
                        link_source_record(conn, tn.source_record_links, sr_id, entity_type, entity_id, created_at, updated_at)

                # ✅ прогресс печатаем без NameError
                done = min(processed, total) if total > 0 else processed
                progress.maybe_print(done, max_seen, every_rows=progress_every_rows, every_secs=progress_every_secs)

                # ✅ translations — пачками
                if (not skip_trans) and len(trans_rows) >= trans_batch_size:
                    progress.pulse(done, max_seen, f"insert trans batch={len(trans_rows)}")
                    with conn.cursor() as cur:
                        cur.executemany(trans_sql, trans_rows)
                    trans_rows.clear()

            # ✅ добить остаток translations
            if (not skip_trans) and trans_rows:
                done = min(processed, total) if total > 0 else processed
                progress.pulse(done, max_seen, f"insert trans batch={len(trans_rows)}")
                with conn.cursor() as cur:
                    cur.executemany(trans_sql, trans_rows)
                trans_rows.clear()

            # flush tags за batch
            if do_tags and batch_slug_to_name and batch_entity_slug_pairs:
                need_slugs = []
                slug_to_id: Dict[str, int] = {}
                for slug in batch_slug_to_name.keys():
                    cached = tag_cache.get(slug)
                    if cached is not None:
                        slug_to_id[slug] = int(cached)
                    else:
                        need_slugs.append(slug)

                if need_slugs:
                    progress.pulse(min(processed, total) if total > 0 else processed, max_seen, f"tags fetch/insert slugs={len(need_slugs)}")
                    existing = bulk_fetch_tag_ids(conn, tags_table, need_slugs, in_limit=tag_in_limit)
                    for s, tid in existing.items():
                        slug_to_id[s] = int(tid)
                        tag_cache.set(s, int(tid))

                    missing = [s for s in need_slugs if s not in existing]
                    if missing:
                        missing_map = {s: batch_slug_to_name[s] for s in missing}
                        bulk_insert_missing_tags(conn, tags_table, missing_map, batch_tags_ca, batch_tags_ua)
                        created = bulk_fetch_tag_ids(conn, tags_table, missing, in_limit=tag_in_limit)
                        for s, tid in created.items():
                            slug_to_id[s] = int(tid)
                            tag_cache.set(s, int(tid))

                pairs: List[Tuple[int, int]] = []
                for eid, slug in batch_entity_slug_pairs:
                    tid = slug_to_id.get(slug)
                    if tid:
                        pairs.append((eid, tid))
                bulk_insert_pivots(conn, pivot_table, pivot_left, pairs, batch_tags_ca, batch_tags_ua)

            conn.commit()
            commits += 1
            last_id = max_seen
            save_checkpoint(source_table, {"last_id": last_id, "updated_at": int(time.time())})

            done = min(processed, total) if total > 0 else processed
            progress.print(done, last_id, status=f"commit={commits}")

        done_final = total if total > 0 else processed
        progress.print(done_final, last_id, status="done", force_newline=True)

    except KeyboardInterrupt:
        try:
            conn.commit()
        except Exception:
            pass
        save_checkpoint(source_table, {"last_id": last_id, "updated_at": int(time.time()), "interrupted": True})
        sys.stdout.write("\n")
        sys.stdout.flush()
        print(f"INTERRUPTED. Saved checkpoint last_id={last_id} for {source_table}")
        raise

    dt = time.time() - t0
    speed = processed / dt if dt > 0 else 0.0
    print(f"OK {kind}: {source_table} processed={processed} in {dt:.1f}s ({speed:.1f} rows/s)")


# -------------------- CLI --------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--locale", default=env("BOOKS_LOCALE", "en"))
    ap.add_argument("--site-id", type=int, default=env("BOOKS_SITE_ID", 0, int))
    ap.add_argument("--category-id", type=int, default=env("BOOKS_CATEGORY_ID", 0, int))
    ap.add_argument("--batch", type=int, default=env("BOOKS_BATCH", 500, int))

    ap.add_argument("--do-covers", type=int, default=env("BOOKS_IMPORT_COVERS", 1, int))
    ap.add_argument("--do-works", type=int, default=env("BOOKS_IMPORT_WORKS", 1, int))
    ap.add_argument("--do-editions", type=int, default=env("BOOKS_IMPORT_EDITIONS", 1, int))
    ap.add_argument("--do-authors", type=int, default=env("BOOKS_IMPORT_AUTHORS", 1, int))
    ap.add_argument("--do-tags", type=int, default=env("BOOKS_IMPORT_TAGS", 1, int))
    ap.add_argument("--do-sources", type=int, default=env("BOOKS_IMPORT_SOURCES", 1, int))
    ap.add_argument("--recalc-tags", type=int, default=env("BOOKS_RECALC_TAGS", 0, int))

    args = ap.parse_args()

    for k in ["MYSQL_DB", "MYSQL_USER", "MYSQL_PASSWORD"]:
        if not env(k):
            print(f"ERROR: missing {k} in .env")
            sys.exit(1)

    conn = db_connect()
    try:
        prefix = detect_table_prefix(conn)
        tn = TableNames(prefix=prefix)
        print(f"Using table prefix: {prefix!r}")

        sources = detect_ol_sources(conn)
        print("Detected OpenLibrary sources:")
        print(f"  work: {sources.get('work', [])}")
        print(f"  edition: {sources.get('edition', [])}")
        print(f"  author: {sources.get('author', [])}")

        if args.do_covers:
            import_covers_metadata(conn, tn)

        if args.do_works:
            for t in sources.get("work", []):
                process_ol_table(conn, tn, t, "work", args.locale, args.site_id, args.category_id, args.batch, bool(args.do_tags), bool(args.do_sources))

        if args.do_editions:
            for t in sources.get("edition", []):
                process_ol_table(conn, tn, t, "edition", args.locale, args.site_id, args.category_id, args.batch, bool(args.do_tags), bool(args.do_sources))

        if args.do_authors:
            for t in sources.get("author", []):
                process_ol_table(conn, tn, t, "author", args.locale, args.site_id, args.category_id, args.batch, bool(args.do_tags), bool(args.do_sources))

        if args.recalc_tags and args.do_tags:
            print("Recalculating tag counts...")
            recalc_tag_counts(conn, tn.tags_books, tn.pivot_books_tags)
            recalc_tag_counts(conn, tn.tags_editions, tn.pivot_editions_tags)
            recalc_tag_counts(conn, tn.tags_authors, tn.pivot_authors_tags)
            print("OK tag counts")

        print("DONE")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
