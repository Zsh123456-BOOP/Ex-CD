# RetriCD Student-Private History Retrieval Task Card

## Goal

Build a compact RetriCD line that is clearly separate from ConceptSkillCDM's CRG/LCRF modules, uses the existing ConceptSkillCDM datasets, and first runs the text-free `full` variant on server GPUs 2 and 3.

## Context

RetriCD should answer a narrower claim than CRG: for each query, the prediction is backed by the same student's query-before interactions. The first version must avoid CRG, cross-student memory, LLM/text retrieval, and student-id embedding shortcuts.

## Constraints

- Use `~/ConceptSkillCDM/data` as the server source and copy datasets into `~/RetriCD/data`.
- Use `zsh@10.154.22.11`.
- Use the server conda env at `/home/zsh/anaconda3/envs/xph_env/bin/python`.
- Use GPUs 2 and 3 for the first `full` run.
- Keep the code small and inspectable.

## Checklist

- [x] Confirm current local RetriCD repo is empty and should become a separate project.
- [x] Confirm ConceptSkillCDM dataset format: `stu_id,exer_id,cpt_seq,label`.
- [x] Confirm server datasets exist for `assist_09`, `assist_17`, `junyi`, and `nips34`.
- [x] Confirm server runtime uses `xph_env` with PyTorch/CUDA rather than system `python3`.
- [x] Create compact RetriCD package.
- [x] Add server data-copy and full-run scripts.
- [x] Run a smoke check under `xph_env`.
- [x] Commit the initial RetriCD code to git.
- [x] Sync code to `zsh@10.154.22.11:~/RetriCD`.
- [x] Copy datasets from `~/ConceptSkillCDM/data` to `~/RetriCD/data`.
- [x] Launch first `full` run on GPUs 2 and 3.
- [x] Record remote log paths and first status check.
- [x] Stop the first incomplete MVP-only full run before replacing the pipeline.
- [x] Split code into the requested RetriCD modules.
- [x] Add cold-handled chronological split builder.
- [x] Add deterministic `nips34_retricd_small` derived dataset path matched to Junyi scale.
- [x] Add weak-evidence analysis and challenge-set tools.
- [x] Add fidelity evaluation for support deletion, same-student random corruption, and shuffle-student.
- [x] Add ablation runner for no/random/term-only retrieval.
- [x] Add parquet predictions/support exports plus JSONL cases.
- [x] Smoke-test the new pipeline under `xph_env`.

## Remote Run

### Stopped MVP Run

- Server: `zsh@10.154.22.11`
- Project directory: `/home/zsh/RetriCD`
- Python: `/home/zsh/anaconda3/envs/xph_env/bin/python`
- Run id: `retricd_full_20260612_163205`
- Launcher PID: `620754`
- Launcher log: `/home/zsh/RetriCD/logs/retricd_full_20260612_163205_launcher.log`
- First wave logs:
  - `/home/zsh/RetriCD/logs/retricd_assist_09_full_20260612_163205.log`
  - `/home/zsh/RetriCD/logs/retricd_assist_17_full_20260612_163205.log`
- First status check: launcher alive after 57 seconds; GPU 2 and GPU 3 both active; `assist_09` reached epoch 2 and `assist_17` completed the first epoch training pass.
- Stop status: terminated before replacing the pipeline; GPU 2 and GPU 3 were confirmed idle afterward.

### Replacement Pipeline Smoke

- Runtime dependency: installed `pyarrow` in `/home/zsh/anaconda3/envs/xph_env/bin/python` for real parquet export.
- Prepared data root: `/home/zsh/RetriCD/data_retricd`
- Source copy root: `/home/zsh/RetriCD/data_source`
- Prepared datasets:
  - `assist_09`: 267,423 rows
  - `assist_17`: 390,281 rows
  - `junyi`: 353,835 rows
  - `nips34_retricd_small`: 353,978 rows, derived from `nips34` by deterministic student-level sampling to match Junyi scale
- Smoke command: one epoch on `assist_09` with GPU 2.
- Smoke outputs:
  - `/home/zsh/RetriCD/outputs/smoke_v2/assist_09/seed42_full/metrics.json`
  - `/home/zsh/RetriCD/outputs/smoke_v2/assist_09/seed42_full/exports/predictions.parquet`
  - `/home/zsh/RetriCD/outputs/smoke_v2/assist_09/seed42_full/exports/supports.parquet`
  - `/home/zsh/RetriCD/outputs/smoke_v2/assist_09/seed42_full/exports/cases.jsonl`
  - `/home/zsh/RetriCD/outputs/analysis_smoke/assist_09/weak_evidence_counts.csv`
  - `/home/zsh/RetriCD/outputs/challenge_smoke/assist_09/test_challenge_sets.csv`

### Complete Full Run

- Run id: `retricd_complete_full_20260612_165433`
- Launcher PID: `634704`
- Launcher log: `/home/zsh/RetriCD/logs/retricd_complete_full_20260612_165433_launcher.log`
- Output root: `/home/zsh/RetriCD/outputs/retricd_complete_full_20260612_165433`
- First wave logs:
  - `/home/zsh/RetriCD/logs/retricd_assist_09_full_20260612_165433.log`
  - `/home/zsh/RetriCD/logs/retricd_assist_17_full_20260612_165433.log`
- First status check: launcher alive after 42 seconds; GPU 2 and GPU 3 active; `assist_09` and `assist_17` both reached epoch 2.
- Planned second wave from the same launcher: `junyi` on GPU 2 and `nips34_retricd_small` on GPU 3.

## MVP Scope

The first `full` model contains:

- query encoder from exercise id, concept set, and train-only difficulty;
- student-private prefix history only;
- interpretable retriever terms: latent similarity, concept overlap, difficulty similarity, and recency;
- top-K support aggregation;
- global prefix summary;
- BCE plus fidelity margin against random same-student support;
- overall and evidence-regime metrics;
- `predictions.csv`, `supports.csv`, and `cases.jsonl` exports.
