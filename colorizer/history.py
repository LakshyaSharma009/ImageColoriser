import sqlite3
import subprocess
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from .config import BASE_DIR, HISTORY_DB_PATH, HISTORY_MAX_ENTRIES, HISTORY_OUTPUT_DIR

# Each function opens its own short-lived connection rather than caching one at module
# scope: Streamlit reruns the script on a fresh ScriptRunner thread per interaction, and a
# cached sqlite3.Connection would eventually be used from a thread other than the one that
# created it, raising sqlite3.ProgrammingError. sqlite3.Connection's own context manager
# only commits/rolls back on exit, it does not close() -- so we wrap it ourselves to avoid
# leaking connections (and, on Windows, leaking file locks on the .db file).


@contextmanager
def _connect(db_path=None):
    path = db_path or HISTORY_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def init_db(db_path=None):
    """Create the `runs` and `experiments` tables if they don't exist yet.

    Uses CREATE TABLE IF NOT EXISTS rather than a migration framework: adding
    a brand-new table is always safe to run against an older database file
    that only has `runs`, so no separate migration step is needed.
    """
    HISTORY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with _connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                model_id TEXT NOT NULL,
                saturation REAL NOT NULL,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                elapsed_s REAL NOT NULL,
                output_path TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                model TEXT NOT NULL,
                dataset TEXT NOT NULL,
                image_count INTEGER NOT NULL,
                saturation REAL NOT NULL,
                device TEXT NOT NULL,
                psnr_mean REAL,
                ssim_mean REAL,
                lpips_mean REAL,
                latency_mean_seconds REAL,
                latency_median_seconds REAL,
                latency_p95_seconds REAL,
                git_commit TEXT
            )
            """
        )


def record_run(*, filename, model_id, saturation, width, height, elapsed_s, output_bytes,
                db_path=None, output_dir=None):
    out_dir = output_dir or HISTORY_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"{uuid.uuid4().hex}.png"
    output_path.write_bytes(output_bytes)

    with _connect(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO runs (filename, model_id, saturation, width, height, elapsed_s, output_path, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                filename,
                model_id,
                saturation,
                width,
                height,
                elapsed_s,
                str(output_path),
                time.strftime("%Y-%m-%d %H:%M:%S"),
            ),
        )
        run_id = cursor.lastrowid

    _prune(db_path)
    return run_id


def list_runs(limit=50, db_path=None):
    with _connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM runs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(row) for row in rows]


def clear_runs(db_path=None):
    with _connect(db_path) as conn:
        conn.execute("DELETE FROM runs")


def _prune(db_path=None, max_entries=None):
    """Keep the history table (and the PNGs it references) bounded in size."""
    cap = max_entries if max_entries is not None else HISTORY_MAX_ENTRIES
    with _connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        overflow = conn.execute(
            "SELECT id, output_path FROM runs ORDER BY id DESC LIMIT -1 OFFSET ?", (cap,)
        ).fetchall()
        if not overflow:
            return
        for row in overflow:
            try:
                if row["output_path"]:
                    Path(row["output_path"]).unlink(missing_ok=True)
            except OSError:
                pass
        conn.executemany("DELETE FROM runs WHERE id = ?", [(row["id"],) for row in overflow])


def _current_git_commit():
    """Best-effort short commit hash for the running checkout, or None outside a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def record_experiment(
    *,
    model,
    dataset,
    image_count,
    saturation,
    device,
    psnr_mean=None,
    ssim_mean=None,
    lpips_mean=None,
    latency_mean_seconds=None,
    latency_median_seconds=None,
    latency_p95_seconds=None,
    git_commit=None,
    db_path=None,
):
    """Record one evaluation/benchmark run (an "experiment") to the history database."""
    if git_commit is None:
        git_commit = _current_git_commit()

    with _connect(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO experiments (
                created_at, model, dataset, image_count, saturation, device,
                psnr_mean, ssim_mean, lpips_mean,
                latency_mean_seconds, latency_median_seconds, latency_p95_seconds,
                git_commit
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                time.strftime("%Y-%m-%d %H:%M:%S"),
                model,
                dataset,
                image_count,
                saturation,
                device,
                psnr_mean,
                ssim_mean,
                lpips_mean,
                latency_mean_seconds,
                latency_median_seconds,
                latency_p95_seconds,
                git_commit,
            ),
        )
        return cursor.lastrowid


def list_experiments(limit=50, db_path=None):
    with _connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM experiments ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(row) for row in rows]


def get_experiment(experiment_id, db_path=None):
    with _connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM experiments WHERE id = ?", (experiment_id,)
        ).fetchone()
        return dict(row) if row else None


def clear_experiments(db_path=None):
    with _connect(db_path) as conn:
        conn.execute("DELETE FROM experiments")
