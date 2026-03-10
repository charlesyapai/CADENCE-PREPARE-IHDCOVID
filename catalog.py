"""
DataCatalog - quick reference
=============================

1.  **Single dataset**

    ```python
    from src.catalog import DataCatalog
    cat = DataCatalog()                 # reads ./catalog.yaml
    df  = cat.load("mediclaims_2015")   # returns a pandas.DataFrame
    ```

2.  **Several datasets at once**

    ```python
    cat.inject_into("mediclaims_2015", "sales_2025", tgt_globals=globals())
    # -> variables called mediclaims_2015 and sales_2025 now exist in your namespace
    # Prints a summary of what was loaded + estimated RAM usage (see below).
    ```

    *Variable names are identical to the alias strings you pass.*  
    If you need custom names, create a mapping:

    ```python
    wanted = {"claims15": "mediclaims_2015", "sales": "sales_2025"}
    dfs = {var: cat.load(alias) for var, alias in wanted.items()}
    ```

3.  **Memory summary**

    Every call to `inject_into()` now prints:

      * a comma-separated list of the DataFrame variables it created
      * the combined size of those DataFrames in MB
      * how much room is left before a soft limit (default = 8 GB, configurable)

---------------------------------------------------------------------------
Public API
---------------------------------------------------------------------------

* `list_categories() -> list[str]`
* `list_datasets(category: str | None = None) -> list[str]`
* `load(alias: str, **pd_kwargs) -> pd.DataFrame`
* `add_dataset(category, alias, *, path, ftype, ...)`
* `inject_into(*aliases, tgt_globals=None, verbose=True, mem_limit_gb=8)`

---------------------------------------------------------------------------
Minimal YAML snippet (for reference)
---------------------------------------------------------------------------
```yaml
mediclaims:
  mediclaims_2015:
    path:  s3://moh-2030/mediclaims2015.csv
    type:  csv
    delimiter: ","
    description: Health insurance claims - calendar year 2015
"""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import s3fs
import yaml
import sys

class DataCatalog:
    """Reads / writes catalog.yaml and loads datasets on demand."""

    def __init__(self, catalog_path: str | Path = "catalog.yaml"):
        self._catalog_file = Path(catalog_path).expanduser()
        if not self._catalog_file.exists():
            raise FileNotFoundError(f"{self._catalog_file} does not exist")
        self._load_yaml()

        # One shared S3 filesystem keeps connections warm
        self._s3 = s3fs.S3FileSystem(anon=False)

    # ------------------------------------------------------------------ #
    # YAML helpers
    # ------------------------------------------------------------------ #
    def _load_yaml(self) -> None:
        self.data: Dict[str, Dict[str, Dict[str, Any]]]
        with self._catalog_file.open() as f:
            self.data = yaml.safe_load(f) or {}

    def _write_yaml(self) -> None:
        with self._catalog_file.open("w") as f:
            yaml.safe_dump(self.data, f, sort_keys=False)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def list_categories(self) -> list[str]:
        return sorted(self.data)

    def list_datasets(self, category: Optional[str] = None) -> list[str]:
        if category:
            return sorted(self.data.get(category, {}))
        # flattened list
        return sorted(
            alias for cat in self.data.values() for alias in cat
        )

    def _find_meta(self, alias: str) -> tuple[str, Dict[str, Any]]:
        hits = [
            (cat, meta)
            for cat, block in self.data.items()
            if (meta := block.get(alias))
        ]
        if not hits:
            raise KeyError(f"Alias '{alias}' not found in catalog")
        if len(hits) > 1:
            cats = [c for c, _ in hits]
            raise ValueError(
                f"Alias '{alias}' exists in multiple categories: {cats}"
            )
        return hits[0]  # (category, meta)

    def load(self, alias: str, **pd_kwargs) -> pd.DataFrame:
        """
        Load dataset as a pandas DataFrame.

        Extra **pd_kwargs override defaults (useful for dtype overrides,
        memory optimisation, etc).
        """
        _cat, meta = self._find_meta(alias)
        path = meta["path"]
        ftype = meta["type"].lower()

        # Decide on file opener: s3fs for s3:// or open() for local
        opener: Any
        if path.startswith("s3://"):
            opener = self._s3.open
        else:
            opener = open

        with opener(path, "rb") as f:
            if ftype == "csv":
                df = pd.read_csv(
                    f,
                    sep=meta.get("delimiter", ","),
                    **pd_kwargs,
                )
            elif ftype == "xlsx":
                df = pd.read_excel(
                    f,
                    sheet_name=meta.get("sheet_name") or 0,
                    engine="openpyxl",
                    **pd_kwargs,
                )
            elif ftype == "parquet":
                # parquet requires a file-like path, let pandas handle s3:// directly
                df = pd.read_parquet(path, **pd_kwargs)
            elif ftype == "jsonl":
                df = pd.read_json(f, lines=True, **pd_kwargs)
            else:
                raise ValueError(f"Unsupported type '{ftype}'")
        return df

    # ------------------------------------------------------------------ #
    # Mutation helpers
    # ------------------------------------------------------------------ #
    def add_dataset(
        self,
        category: str,
        alias: str,
        *,
        path: str,
        ftype: str,
        delimiter: str | None = None,
        sheet_name: str | None = None,
        description: str = "",
        overwrite: bool = False,
        columns: dict[str,str]|None=None,
        required_columns: list[str]|None=None, **extra) -> None:
        """Add or overwrite a catalog entry programmatically."""
        block = self.data.setdefault(category, {})
        if alias in block and not overwrite:
            raise ValueError(
                f"{alias} already exists under '{category}'. Use overwrite=True."
            )

        meta = dict(
            path=path,
            type=ftype,
            delimiter=delimiter,
            sheet_name=sheet_name,
            description=description,
        )
        if columns:           meta["columns"] = columns
        if required_columns:  meta["required_columns"] = required_columns
        meta.update(extra)    # future-proof
        
        # Strip Nones to keep YAML clean
        meta = {k: v for k, v in meta.items() if v is not None}
        block[alias] = meta
        self._write_yaml()

    # ------------------------------------------------------------------ #
    # Convenience
    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #
    # Convenience
    # ------------------------------------------------------------------ #
    def inject_into(
        self,
        *aliases: str,
        tgt_globals: dict | None = None,
        verbose: bool = True,
        mem_limit_gb: int = 8,
    ):
        """
        Load each *alias* and bind the DataFrame into *tgt_globals*
        (defaults to the caller's globals()).

        Example
        -------
        >>> cat.inject_into("sales_2025", "claims_2015", tgt_globals=globals())

        Parameters
        ----------
        verbose : bool, default True
            Print a summary of variables loaded and memory usage.
        mem_limit_gb : int, default 8
            Soft memory budget; the summary shows how much headroom remains.
        """
        g = tgt_globals if tgt_globals is not None else globals()

        loaded_vars: list[str] = []
        total_bytes = 0

        for alias in aliases:
            df = self.load(alias)
            g[alias] = df          # variable name == alias
            loaded_vars.append(alias)
            total_bytes += df.memory_usage(deep=True).sum()

        if verbose:
            used_mb = total_bytes / 1024 ** 2
            remaining_mb = mem_limit_gb * 1024 - used_mb
            print(
                f"[DataCatalog] Loaded {', '.join(loaded_vars)} "
                f"({used_mb:,.1f} MB).  Approx. head-room: {remaining_mb:,.1f} MB "
                f"(limit {mem_limit_gb} GB)."
            )
    # ------------------------------------------------------------------ #
    # Stringification
    # ------------------------------------------------------------------ #
    def __repr__(self) -> str:
        return (
            f"<DataCatalog {self._catalog_file} "
            f"({len(self.list_categories())} categories, "
            f"{len(self.list_datasets())} datasets)>"
        )

# ---------------------------------------------------------------------- #
# Mini-demo when run directly: python catalog.py [optional_path]
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    # Check if a path was provided in the command, else default
    path = sys.argv[1] if len(sys.argv) > 1 else "catalog.yaml"

    try:
        cat = DataCatalog(path)
        print(f"Using catalog: {path}")
        print(cat)
        print("Categories:", cat.list_categories())
        for ds in cat.list_datasets()[:5]:
            print(" .", ds)
    except FileNotFoundError:
        print(f"Error: The catalog file '{path}' was not found.")
        sys.exit(1)