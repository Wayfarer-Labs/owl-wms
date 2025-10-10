import json
import sqlite3
import threading
import uuid
import numpy as np
from pathlib import Path
from typing import List, Any


class NpyTable:
    def __init__(self, directory: str, columns: List[str] = None, array_columns: set[str] = None, primary_key: str = None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

        # SQLite: single place for schema + rows
        self.db_path = self.directory / "manifest.sqlite3"
        self._db = sqlite3.connect(self.db_path, check_same_thread=False)
        self._db.executescript("""
            PRAGMA journal_mode=WAL;
            PRAGMA synchronous=NORMAL;
            CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS manifest (row_json TEXT NOT NULL);
        """)
        # Load or initialize schema in DB
        row = self._db.execute("SELECT value FROM meta WHERE key='schema'").fetchone()
        if row:
            schema = json.loads(row[0])
            if columns is not None and columns != schema["columns"]:
                raise AssertionError("columns mismatch")
            if array_columns is not None and set(array_columns) != set(schema["array_columns"]):
                raise AssertionError("array_columns mismatch")
            if primary_key is not None and primary_key != schema.get("primary_key"):
                raise AssertionError("primary_key mismatch")
        else:
            if not primary_key:
                raise AssertionError("primary_key is required")
            if columns is not None and primary_key not in columns:
                raise AssertionError("primary_key must be one of columns")
            if columns is not None and array_columns is not None:
                if not set(array_columns).issubset(set(columns)):
                    raise AssertionError("array_columns must be a subset of columns")
            schema = {
                "columns": columns or [],  # allow caller to supply exact list
                "array_columns": list(array_columns or []),
                "primary_key": primary_key,
            }
            with self._db:
                self._db.execute("INSERT INTO meta(key, value) VALUES('schema', ?)", (json.dumps(schema),))
        self.columns = schema["columns"]
        self.array_columns = set(schema["array_columns"])
        self.primary_key = schema.get("primary_key")

        # Ensure uniqueness of the primary key inside row_json
        if self.primary_key:
            self._db.execute(
                f'CREATE UNIQUE INDEX IF NOT EXISTS manifest_pk '
                f'ON manifest(json_extract(row_json, \'$."{self.primary_key}"\'))'
            )

    # TODO: @classmethod `def create(...)` to initialize

    def __len__(self):
        with self._lock:
            return self._db.execute("SELECT COUNT(*) FROM manifest").fetchone()[0]

    def append(self, **row: Any) -> int:
        # must be exact keys
        if set(row) != set(self.columns):
            raise ValueError(f"Expected columns {self.columns}, got {list(row)}")

        entry = {}
        for key, val in row.items():
            if key in self.array_columns:
                uid = uuid.uuid4().hex
                final_path = self.directory / f"{key}_{uid}.npy"
                tmp_path = final_path.with_suffix(".npy.tmp")
                arr = np.asarray(val, order="C")
                with open(tmp_path, "wb", buffering=8 << 20) as f:  # 8 MiB buffer
                    np.save(f, arr, allow_pickle=False)
                tmp_path.replace(final_path)  # atomic publish
                entry[key] = final_path.name
            else:
                entry[key] = val

        # Atomic insert under a lock; 0-based idx is (rowid - 1)
        with self._lock, self._db:
            try:
                cur = self._db.execute("INSERT INTO manifest(row_json) VALUES (?)", (json.dumps(entry),))
            except sqlite3.IntegrityError as e:
                raise ValueError("Duplicate primary key") from e
            return cur.lastrowid - 1

    def add_column(self, name: str, values, array: bool = False):
        if name in self.columns:
            raise ValueError(f"Column {name!r} already exists")
        try:
            vals = list(values)  # accept any iterable
        except TypeError as e:
            raise TypeError("values must be an iterable") from e
        # ---- Phase 1: take a stable snapshot under the lock ----
        with self._lock:
            baseline_len = self._db.execute("SELECT COUNT(*) FROM manifest").fetchone()[0]
        if len(vals) != baseline_len:
            raise ValueError(f"Expected {baseline_len} values, got {len(vals)}")

        # ---- Phase 2: do the heavy I/O without holding the lock ----
        # Pre-write array blobs (tmp -> replace) so they are ready before publication.
        # Store the final file names we will reference in the manifest.
        file_names: list[str | None] = [None] * baseline_len
        if array:
            for i, val in enumerate(vals):
                final_path = self.directory / f"{name}_{i}.npy"
                tmp_path = final_path.with_suffix(".npy.tmp")
                with open(tmp_path, "wb", buffering=8 << 20) as f:
                    np.save(f, np.asarray(val, order="C"), allow_pickle=False)
                tmp_path.replace(final_path)  # atomic publish of the blob
                file_names[i] = final_path.name

        # ---- Phase 3: publish atomically under the lock ----
        with self._lock, self._db:
            # Abort if table changed between phases (e.g., append happened).
            if self._db.execute("SELECT COUNT(*) FROM manifest").fetchone()[0] != baseline_len:
                raise RuntimeError("Table changed during add_column; retry the operation")

            # Update schema in DB (meta table)
            self.columns.append(name)
            if array:
                self.array_columns.add(name)
            schema = {
                "columns": self.columns,
                "array_columns": list(self.array_columns),
                "primary_key": self.primary_key,
            }
            self._db.execute("UPDATE meta SET value=? WHERE key='schema'", (json.dumps(schema),))

            # Update each row's JSON payload (ordered by rowid)
            rows = list(self._db.execute("SELECT rowid, row_json FROM manifest ORDER BY rowid"))
            for i, (rowid, row_json) in enumerate(rows):
                entry = json.loads(row_json)
                entry[name] = file_names[i] if array else vals[i]
                self._db.execute("UPDATE manifest SET row_json=? WHERE rowid=?", (json.dumps(entry), rowid))

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.get(columns=[key])[0]
        elif isinstance(key, (list, tuple)):
            return self.get(columns=list(key))
        else:
            raise KeyError(f"Invalid key: {key!r}")

    def get(self, columns: List[str], rows: List[int] | None = None) -> List[List[Any]]:
        with self._lock:
            array_cols = set(self.array_columns)
            directory = self.directory
            columns_snapshot = list(self.columns)

        invalid = set(columns) - set(columns_snapshot)
        if invalid:
            raise KeyError(f"Unknown columns requested: {invalid}")

        # Project only requested columns via JSON1; optionally restrict to given rows.
        extracts = ", ".join(f"json_extract(row_json, '$.\"{c}\"')" for c in columns)
        base_sql = f"SELECT rowid, {extracts} FROM manifest"
        with self._lock:
            if rows is None:
                ordered = self._db.execute(base_sql + " ORDER BY rowid").fetchall()
            else:
                rowids = [r + 1 for r in rows]  # Python 0-based → SQLite rowid 1-based
                q = base_sql + f" WHERE rowid IN ({','.join('?' for _ in rowids)})"
                fetched = self._db.execute(q, tuple(rowids)).fetchall()
                m = {rid: rec for rid, *rec in fetched}
                ordered = [(r + 1, *m[r + 1]) for r in rows]  # preserve caller order

        def materialize(i: int, col: str) -> List[Any]:
            vs = [rec[i] for rec in ordered]  # i=1.. since col0 is rowid
            if col in array_cols:
                return [np.load(directory / v, mmap_mode="r") for v in vs]
            return [json.loads(v) if isinstance(v, str) and v[:1] in ("[", "{") else v for v in vs]

        return [materialize(i, col) for i, col in enumerate(columns, 1)]
