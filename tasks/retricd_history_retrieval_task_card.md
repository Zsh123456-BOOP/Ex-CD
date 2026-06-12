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

## Remote Run

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
