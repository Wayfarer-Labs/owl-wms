import json
import threading
import uuid
import numpy as np
from pathlib import Path
from typing import List, Any


class NpyTable:
    # required fields per row
    default_columns = [
        "video", "audio", "mouse", "buttons",
        "tarball", "pt_idx", "missing", "truncated", "seq_len"
    ]
    # ndarray blobs
    default_array_columns = {"video", "audio", "mouse", "buttons"}

    def __init__(self, directory: str, columns: List[str] | None = None, array_columns: set[str] | None = None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

        # set schema / ensure consistent with existing schema
        self.schema_path = self.directory / "schema.json"
        if self.schema_path.exists():
            schema = json.loads(self.schema_path.read_text())
            assert columns is None or columns == schema["columns"], "columns mismatch"
            assert (
                array_columns is None or set(array_columns) == set(schema["array_columns"])
            ), "array_columns mismatch"
            columns = schema["columns"]
            array_columns = schema["array_columns"]
        else:
            columns = columns or self.default_columns
            array_columns = array_columns or list(self.default_array_columns)
            self.schema_path.write_text(
                json.dumps({"columns": columns, "array_columns": array_columns})
            )
        self.columns = columns
        self.array_columns = set(array_columns)

        self.manifest_path = self.directory / "manifest.json"
        if self.manifest_path.exists():
            self.manifest = json.loads(self.manifest_path.read_text())
        else:
            self.manifest = []

    def __len__(self):
        return len(self.manifest)

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

        # Atomic manifest update under a lock
        with self._lock:
            idx = len(self.manifest)          # position this row will take
            self.manifest.append(entry)
            tmp = self.manifest_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(self.manifest))
            tmp.replace(self.manifest_path)   # atomic publish
        return idx

    def add_column(self, name: str, values, array: bool = False):
        if name in self.columns:
            raise ValueError(f"Column {name!r} already exists")
        try:
            vals = list(values)  # accept any iterable
        except TypeError as e:
            raise TypeError("values must be an iterable") from e
        # ---- Phase 1: take a stable snapshot under the lock ----
        with self._lock:
            baseline_len = len(self.manifest)
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
        with self._lock:
            # Abort if table changed between phases (e.g., append happened).
            if len(self.manifest) != baseline_len:
                raise RuntimeError("Table changed during add_column; retry the operation")

            # Update schema in-memory
            self.columns.append(name)
            if array:
                self.array_columns.add(name)

            # Update manifest in-memory
            for i, entry in enumerate(self.manifest):
                if array:
                    entry[name] = file_names[i]  # already written
                else:
                    entry[name] = vals[i]

            manifest_tmp = self.manifest_path.with_suffix(".json.tmp")
            manifest_tmp.write_text(json.dumps(self.manifest))
            manifest_tmp.replace(self.manifest_path)

            schema_tmp = self.schema_path.with_suffix(".json.tmp")
            schema_tmp.write_text(json.dumps({"columns": self.columns,
                                              "array_columns": list(self.array_columns)}))
            schema_tmp.replace(self.schema_path)

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.get(columns=[key])[0]
        elif isinstance(key, (list, tuple)):
            return self.get(columns=list(key))
        else:
            raise KeyError(f"Invalid key: {key!r}")

    def get(self, columns: List[str], rows: List[int] | None = None) -> List[List[Any]]:
        with self._lock:
            manifest_snapshot = [e.copy() for e in self.manifest]
            array_cols = set(self.array_columns)
            directory = self.directory
            columns_snapshot = list(self.columns)

        invalid = set(columns) - set(columns_snapshot)
        if invalid:
            raise KeyError(f"Unknown columns requested: {invalid}")

        rows = range(len(manifest_snapshot)) if rows is None else rows

        return [
            [
                np.load(directory / manifest_snapshot[r][col], mmap_mode="r")
                if col in array_cols
                else manifest_snapshot[r][col]
                for r in rows
            ]
            for col in columns
        ]
