# OpenLibrary Dump Import — Complete Guide

This document describes a **clean Python installation**, creation of a virtual environment, dependency setup, and running the TXT dump import script into MySQL.

The guide is intended for **Ubuntu / Debian / Linux servers**.

---

## 1. Install Python

Check whether Python is installed:

```bash
python3 --version
```

If the command is not found, install it:

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip
```

Verify:

```bash
python3 --version
pip3 --version
```

Expected output:

```bash
Python 3.10+
pip 22+
```

---

## 2. Create a Working Directory

For example:

```bash
mkdir -p ~/tmp/import_openlibrary
cd ~/tmp/import_openlibrary
```

Copy the following into this directory:

- `import_load.py`
- `.env`
- all `ol_dump_*.txt` files

If the dumps are missing, download them (latest dumps are available at https://openlibrary.org/developers/dumps):

```bash
wget https://openlibrary.org/data/ol_dump_editions_latest.txt.gz
wget https://openlibrary.org/data/ol_dump_works_latest.txt.gz
wget https://openlibrary.org/data/ol_dump_authors_latest.txt.gz
wget https://openlibrary.org/data/ol_dump_ratings_latest.txt.gz
wget https://openlibrary.org/data/ol_dump_reading-log_latest.txt.gz
wget https://openlibrary.org/data/ol_dump_redirects_latest.txt.gz
wget https://openlibrary.org/data/ol_dump_deletes_latest.txt.gz
wget https://openlibrary.org/data/ol_dump_lists_latest.txt.gz
wget https://openlibrary.org/data/ol_dump_other_latest.txt.gz
wget https://openlibrary.org/data/ol_dump_covers_metadata_latest.txt.gz
wget https://openlibrary.org/data/ol_dump_wikidata_latest.txt.gz
```

Extract them in place:

```bash
gunzip *.gz
```

---

## 3. Create a Virtual Environment (REQUIRED)

Do not install packages globally — this may break your system.

Create a venv:

```bash
python3 -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

After activation, the prompt should begin with:

```bash
(.venv)
```

Verify:

```bash
which python
```

Expected:

```bash
.../.venv/bin/python
```

---

## 4. Install Dependencies

Only one package is required:

```bash
pip install pymysql
```

Verify:

```bash
python -c "import pymysql; print('OK')"
```

---

## 5. Configure MySQL

Enable LOCAL INFILE.

Inside MySQL:

```bash
SET GLOBAL local_infile = 1;
```

Check:

```bash
SHOW VARIABLES LIKE 'local_infile';
```

Expected:

```bash
ON
```

### Allow it in the config (if it switches OFF after restart)

Open:

```bash
/etc/mysql/mysql.conf.d/mysqld.cnf
```

Add inside `[mysqld]`:

```bash
local_infile=1
```

Restart MySQL:

```bash
sudo systemctl restart mysql
```

---

## 6. Configure `.env`

Example:

```bash
MYSQL_DB=openlibrary
MYSQL_USER=root
MYSQL_PASSWORD=SUPER_PASSWORD
MYSQL_HOST=127.0.0.1
MYSQL_PORT=3306

DUMPS_DIR=/home/user/tmp
TMP_TSV_DIR=/home/user/tmp/ol_tmp_tsv

CHUNK_LINES=300000
PROGRESS_EVERY_CHUNKS=1
CREATE_DB=1
```

### Important about `TMP_TSV_DIR`

These are temporary files used for `LOAD DATA`.

They may consume tens of gigabytes but are deleted after loading.

Prefer placing them on an SSD.

---

## 7. Verify Dumps

The directory should contain:

```bash
ol_dump_authors_latest.txt
ol_dump_works_latest.txt
ol_dump_editions_latest.txt
...
```

Check:

```bash
ls -lh ol_dump_*.txt
```

---

## 8. Run the Import

Activate the virtual environment:

```bash
source .venv/bin/activate
```

Run:

```bash
python import_load.py
```

Expected output:

```bash
FILES=11

=== ol_dump_authors_latest ===
written=...

=== ol_dump_works_latest ===
written=...
```

---

## 9. Run Inside tmux (Recommended)

The import may take many hours.

Create a session:

```bash
tmux new -s import
```

Run the script.

Detach:

```bash
CTRL+B → D
```

Return:

```bash
tmux attach -t import
```

---

## 10. If the Import Stops

The script supports resume using byte offsets.

It will continue from where it stopped.

Simply run again:

```bash
python import_load.py
```

---

## 11. How to Re-import a File

⚠️ Only if absolutely necessary.

```sql
USE openlibrary;

DELETE FROM import_state
WHERE table_name='ol_dump_wikidata_latest';

TRUNCATE TABLE ol_dump_wikidata_latest;
```

Then run again:

```bash
python import_load.py
```

---

## 12. Performance Expectations

`LOAD DATA` typically achieves:

```
30k – 150k rows/sec
```

Depends on:

- SSD vs HDD  
- CPU  
- InnoDB buffer size  
- JSON size  

---

## 13. Common Errors

### pymysql not installed

Solution:

```bash
source .venv/bin/activate
pip install pymysql
```

### local_infile OFF

Enable:

```bash
SET GLOBAL local_infile=1;
```

### python command not found

Use:

```bash
python3
```

Or work inside the virtual environment.

### Too many open files

Increase the limit:

```bash
ulimit -n 65535
```

---

## 14. Performance Recommendations

Increase the InnoDB buffer:

```bash
innodb_buffer_pool_size = 8G
```

(Example for a server with 32GB RAM.)

Disable binary logs if replication is not required — this significantly speeds up imports.

Do not run imports from a network drive.

Use a local SSD only.

---

## 15. Exit the Virtual Environment

```bash
deactivate
```
