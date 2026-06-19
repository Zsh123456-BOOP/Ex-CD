# Ex-CD: Exposure-Aware Debiased Cognitive Diagnosis

Ex-CD is a lightweight, text-free cognitive-diagnosis line. Its claim: **aggregate DOA hides
where the per-concept mastery artifact is unreliable.** Under MNAR exposure (students practice a
skewed subset of concepts), mastery on rarely-practiced concepts is silently over/under-confident,
yet a single aggregate DOA stays high because it is dominated by high-exposure pairs.

Two contributions:
1. **Exposure-Stratified DOA** — DOA + monotonicity bucketed by per-concept train-exposure decile,
   reporting the head-minus-tail gap. Shows DOA itself is MNAR-biased (the Figure-1 cliff).
2. **A non-generative debiaser** — a closed-form per-concept exposure-propensity reweighting (IPS)
   and a doubly-robust (DR) variant bolted onto NeuralCD/KaNCD. No VAE, no LLM, no graph.

This replaces the abandoned RetriCD line (see `tasks/excd_exposure_debias_task_card.md` for the
full plan and the reasons RetriCD was dropped).

## Layout

```
excd/
  data.py        # CSV splits -> id maps, Q-matrix, per-concept exposure propensity + deciles
  model.py       # NCDM + KaNCD backbones (monotone net, per-concept mastery readout)
  losses.py      # vanilla BCE / IPS-reweighted BCE / doubly-robust
  metrics.py     # AUC/ACC/RMSE/BCE, DOA, Exposure-Stratified DOA
  train.py       # run(cfg) core + CLI runner (per-epoch logging, metric/DOA export)
  mnar_stress.py # Phase-2: synthetic exposure-skew sweep (vanilla vs ips tail-gap)
  summarize.py   # aggregate metrics.json -> one CSV table
configs/         # per-dataset YAML
scripts/         # server run scripts (tee logs to logs/)
data_retricd/    # prepared text-free splits (stu_id,exer_id,cpt_seq,label) — REUSED, do not delete
```

## Quick start

```bash
pip install -r requirements.txt

# single run
python -m excd.train --dataset assist_09 --model ncdm --variant vanilla --data-root data_retricd --output-dir outputs
python -m excd.train --dataset assist_09 --model ncdm --variant ips     --data-root data_retricd --output-dir outputs

# or via config
python -m excd.train --config configs/excd_assist09.yaml --variant ips
```

## Full sweep on the server (with logs)

```bash
# args: PYTHON DEVICE   (e.g. the server env + a GPU)
bash scripts/run_excd_all.sh /home/zsh/anaconda3/envs/xph_env/bin/python cuda:2
bash scripts/run_excd_mnar_stress.sh /home/zsh/anaconda3/envs/xph_env/bin/python cuda:3
```

`run_excd_all.sh` sweeps `{assist_09, assist_17, junyi, nips34_retricd_small} × {ncdm} ×
{vanilla, ips}`, streams each run to `logs/excd_<dataset>_<model>_<variant>_<ts>.log`, writes
per-run outputs under `outputs/excd_<ts>/`, and emits `outputs/excd_<ts>/summary.csv`.

## What to read after a run

Per run, `outputs/.../<model>_<variant>_seed42/`:
- `metrics.json` — test AUC/ACC/RMSE/BCE, aggregate **DOA**, and **exposure_stratified_doa**
  (`per_decile`, `bottom_group_doa` [deciles 0-2], `top_group_doa` [deciles 7-9], `group_gap`,
  `n_measurable_deciles`).
- `history.csv` — per-epoch train loss + valid metrics.
- `concept_doa.csv` — per-concept DOA + exposure (re-plot the cliff without re-running).

**Reading the result (the closed loop):**
- *Figure-1 (phenomenon):* in the **vanilla** run, `per_decile` DOA should fall from the top group
  (deciles 7-9) toward chance at the bottom group (deciles 0-2) while aggregate DOA stays high.
  That is the cliff. (Use the **group** gap, not single deciles — the rarest decile is often too
  data-poor to measure; `n_measurable_deciles` flags this.)
- *Method:* **ips**/**dr** should **shrink `group_gap`** vs vanilla while keeping aggregate AUC/DOA
  roughly flat. The tail-gap shrink is the headline, not aggregate AUC.
- *Negative control:* on near-uniform-exposure dense sets the gap should be small and ips≈vanilla.
- *MNAR-stress:* `outputs/mnar_stress/<dataset>_<model>_stress_summary.csv` — vanilla's gap grows
  with gamma; ips should grow more slowly.

Pull the `logs/`, `summary.csv`, and `metrics.json` files back for evaluation.
