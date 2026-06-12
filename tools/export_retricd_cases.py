#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        try:
            return pd.read_parquet(path)
        except Exception:
            csv_path = path.with_suffix(".csv")
            if csv_path.exists():
                return pd.read_csv(csv_path)
            raise
    return pd.read_csv(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Join prediction/support exports into a compact case table.")
    parser.add_argument("--export-dir", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()
    export_dir = Path(args.export_dir)
    preds = _read_table(export_dir / "predictions.parquet")
    supports = _read_table(export_dir / "supports.parquet")
    cases = preds.sort_values("hist_len").head(args.limit).merge(supports, on="row_id", how="left")
    out = Path(args.out) if args.out else export_dir / "case_table.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    cases.to_csv(out, index=False)
    print(f"[OK] wrote {out}")


if __name__ == "__main__":
    main()

