# CertCD: Instance-Level Identifiability Certificates with Calibrated Abstention for Cognitive Diagnosis

CertCD turns the hard information limit of cognitive diagnosis into the contribution. Our own
experiments showed that on standard text-free benchmarks, per-concept mastery is mostly
unidimensional student ability plus a weak concept-specific signal that **collapses to chance on
rarely-practiced concepts** — an information limit no debiasing method can overcome. Instead of
pretending to recover absent signal, CertCD **certifies, per (student, concept) cell, whether that
mastery coordinate is statistically identifiable** given the student's realized item coverage and
the Q-matrix structure, then **abstains** on non-identifiable cells and reports calibrated
reliability.

It is cognitive diagnosis (it produces and validates a per-concept mastery artifact via DOA), not
knowledge tracing.

## The claims and how they are tested

1. **The certificate predicts where mastery is trustworthy** — *synthetic validation* (primary
   novelty evidence, non-circular): with known true mastery, the certificate (from coverage
   geometry + Q only) predicts the cells where the *learned* mastery is actually correct, and does
   so better than a coverage-count baseline. (`certcd/run_synthetic.py`)
2. **Selective prediction kill-switch** — on real data, using the certificate as an abstention
   score must yield **lower AURC than BOTH the count baseline AND a learned-confidence (MC-dropout)
   baseline** (and margin/ability/random). If it does not beat count AND mc_dropout, the
   identifiability machinery is not pulling its weight — the run reports `kill_switch_pass`.
   (`certcd/run.py`)
3. **Calibration concentrates where the certificate says** — ECE is reported on certified vs
   abstained cells (abstained should be worse). (`certcd/run.py`)
4. **Decision task** — certificate-guided adaptive item selection certifies a student's profile in
   fewer items than random / popularity / max-discrimination. (`certcd/cat.py`)

## Layout

```
excd/                 # validated backbone library (REUSED): CDM (NCDM/KaNCD), data, DOA metrics
certcd/
  certificate.py      # per-(student,concept) identifiability certificate (coverage geometry + Q + disc)
  scores.py           # abstention scores: certificate, count, mc_dropout, prob_margin, ability, random
  calibrate.py        # temperature scaling + ECE (overall / certified / abstained)
  selective.py        # risk-coverage, AURC, selective-AUC, excess-AURC (the kill-switch)
  backbone.py         # stable CDM trainer (reuses excd; monotonicity clipper + grad clip)
  synthetic.py        # DINA-style generator with known true mastery + controllable Q/coverage
  run.py              # real-data pipeline (CLI)
  run_synthetic.py    # synthetic certificate-precision/recall validation (CLI)
  cat.py              # CAT certification-efficiency decision task (CLI)
  summarize.py        # aggregate metrics.json -> kill-switch dashboard CSV
configs/certcd_*.yaml
scripts/run_certcd_all.sh
data_retricd/         # prepared text-free splits (stu_id,exer_id,cpt_seq,label) — REUSED
```

## Quick start

```bash
pip install -r requirements.txt
python -m certcd.run --dataset assist_09 --model ncdm --data-root data_retricd --output-dir outputs
python -m certcd.run_synthetic --output-dir outputs/synth
python -m certcd.cat --dataset assist_09 --data-root data_retricd --output-dir outputs/cat
```

## Full run on the server (with logs)

```bash
bash scripts/run_certcd_all.sh /home/zsh/anaconda3/envs/xph_env/bin/python cuda:1
```

Sweeps `{assist_09, assist_17, junyi, nips34_retricd_small} × {ncdm, kancd}` for selective
prediction, runs synthetic validation and CAT, and writes `outputs/certcd_<ts>/certcd_summary.csv`.

## Reading the result (the go/no-go)

- **Synthetic** (`outputs/.../synth/synthetic_*.json`): `cert_auroc_attempted` should exceed
  `count_auroc_attempted`, and `acc_certified` ≫ `acc_abstained`. This is the core evidence.
- **Real kill-switch** (`certcd_summary.csv`): `excess_vs_count > 0` AND `excess_vs_mc_dropout > 0`
  (i.e. `kill_switch_pass = True`) on the concept-structured datasets (assist_09/17, nips34).
  If the certificate ties or loses to count/mc_dropout on real data, the contribution is in
  trouble — reconsider before scaling.
- **Calibration**: `ece_abstained > ece_certified` (uncertainty is where the certificate says).
- **CAT** (`outputs/.../cat/*.json`): `certificate_greedy` items-to-certify < the heuristics.

junyi is a degenerate edge case (concept == exercise); do not read its certificate as a real
hierarchy result.
