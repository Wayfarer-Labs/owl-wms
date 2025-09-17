import json
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

        idx = len(self.manifest)
        entry = {}
        for key, val in row.items():
            if key in self.array_columns:
                path = self.directory / f"{key}_{idx}.npy"
                arr = np.asarray(val, order="C")
                with open(path, "wb", buffering=8 << 20) as f:  # 8 MiB buffer
                    np.save(f, arr, allow_pickle=False)
                entry[key] = f"{key}_{idx}.npy"
            else:
                entry[key] = val

        self.manifest.append(entry)
        self.manifest_path.write_text(json.dumps(self.manifest))
        return idx

    def add_column(self, name: str, values, array: bool = False):
        if name in self.columns:
            raise ValueError(f"Column {name!r} already exists")
        try:
            vals = list(values)  # accept any iterable
        except TypeError as e:
            raise TypeError("values must be an iterable") from e
        if len(vals) != len(self):
            raise ValueError(f"Expected {len(self)} values, got {len(vals)}")

        self.columns.append(name)
        if array:
            self.array_columns.add(name)

        for i, (entry, val) in enumerate(zip(self.manifest, vals)):
            if array:
                path = self.directory / f"{name}_{i}.npy"
                with open(path, "wb", buffering=8 << 20) as f:
                    np.save(f, np.asarray(val, order="C"), allow_pickle=False)
                entry[name] = path.name
            else:
                entry[name] = val

        self.schema_path.write_text(json.dumps({"columns": self.columns,
                                                "array_columns": list(self.array_columns)}))
        self.manifest_path.write_text(json.dumps(self.manifest))

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.get(columns=[key])[0]
        elif isinstance(key, (list, tuple)):
            return self.get(columns=list(key))
        else:
            raise KeyError(f"Invalid key: {key!r}")

    def get(self, columns: List[str], rows: List[int] | None = None) -> List[List[Any]]:
        invalid = set(columns) - set(self.columns)
        if invalid:
            raise KeyError(f"Unknown columns requested: {invalid}")

        rows = range(len(self.manifest)) if rows is None else rows

        return [
            [
                np.load(self.directory / self.manifest[r][col], mmap_mode="r")
                if col in self.array_columns
                else self.manifest[r][col]
                for r in rows
            ]
            for col in columns
        ]
