"""Aggregate CertCD real-data metrics.json files into one CSV (the kill-switch dashboard).

    python -m certcd.summarize --output-dir outputs --out outputs/certcd_summary.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os

COLUMNS = [
    "dataset", "model", "auc", "doa",
    "temperature", "ece_uncalibrated", "ece_calibrated", "ece_certified", "ece_abstained",
    "certified_prediction_fraction",
    "aurc_certificate", "aurc_count", "aurc_mc_dropout", "aurc_margin",
    "excess_vs_count", "excess_vs_mc_dropout", "excess_vs_margin", "excess_vs_ability",
    "kill_switch_pass",
]


def collect(output_dir):
    rows = []
    for path in sorted(glob.glob(os.path.join(output_dir, "**", "metrics.json"), recursive=True)):
        try:
            m = json.load(open(path))
        except Exception:
            continue
        if "selective" not in m:  # skip non-certcd metrics.json (e.g. excd runs)
            continue
        pm = m["selective"]["per_method"]
        ex = m["selective"]["certificate_excess_aurc_vs"]
        cal = m.get("calibration", {})
        rows.append({
            "dataset": m.get("dataset"), "model": m.get("model"),
            "auc": m["test"]["auc"], "doa": m.get("doa", {}).get("aggregate"),
            "temperature": cal.get("temperature"),
            "ece_uncalibrated": cal.get("ece_uncalibrated"), "ece_calibrated": cal.get("ece_calibrated"),
            "ece_certified": cal.get("ece_certified"), "ece_abstained": cal.get("ece_abstained"),
            "certified_prediction_fraction": m.get("certificate", {}).get("certified_prediction_fraction"),
            "aurc_certificate": pm.get("certificate", {}).get("aurc"),
            "aurc_count": pm.get("count", {}).get("aurc"),
            "aurc_mc_dropout": pm.get("mc_dropout", {}).get("aurc"),
            "aurc_margin": pm.get("prob_margin", {}).get("aurc"),
            "excess_vs_count": ex.get("count"), "excess_vs_mc_dropout": ex.get("mc_dropout"),
            "excess_vs_margin": ex.get("prob_margin"), "excess_vs_ability": ex.get("ability"),
            "kill_switch_pass": m.get("kill_switch_pass"),
        })
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=str, default="outputs")
    p.add_argument("--out", type=str, default="outputs/certcd_summary.csv")
    args = p.parse_args()
    rows = collect(args.output_dir)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {len(rows)} rows -> {args.out}")
    for r in rows:
        print(f"  {r['dataset']:<22} {r['model']:<6} AURC_cert={r['aurc_certificate']} "
              f"excess(count={r['excess_vs_count']}, mcdo={r['excess_vs_mc_dropout']}) "
              f"kill_switch={r['kill_switch_pass']}")


if __name__ == "__main__":
    main()
