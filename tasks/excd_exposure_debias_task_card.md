# Ex-CD: Exposure-Aware Debiased Cognitive Diagnosis — Implementation & Experiment Plan

> Supersedes the RetriCD thesis (abandoned 2026-06-19: not novel vs AKT/MACD, reads as KT,
> own fidelity experiments refuted personalization — shuffle_student ≈ full). RetriCD code is
> retired in this repo; the old implementation and outputs remain recoverable from Git history.

## 0. Claim, contribution, and honest scope

**Claim.** Aggregate DOA hides where the per-concept mastery artifact is unreliable. Under MNAR
exposure (students practice a skewed subset of concepts), mastery on **rarely-practiced
concepts** is silently over/under-confident, yet the single aggregate DOA stays high because it is
dominated by high-exposure pairs.

**Two contributions:**
1. **Evaluation axis (the flag nobody planted):** *Exposure-Stratified DOA* + monotonicity,
   showing DOA/monotonicity are themselves MNAR-biased. No listed 2024–26 paper does this.
2. **Method (lightweight, non-generative):** a closed-form per-concept exposure-propensity
   reweighting (IPS) + a doubly-robust (DR) variant bolted onto NeuralCD/KaNCD. Order-of-magnitude
   cheaper than DBCD's β-VAE; backbone-agnostic.

**Venue ceiling (honest):** novelty rated *incremental-but-defensible*. Realistic target = A-tier
**applied** track / EDM / LAK / datasets-and-benchmarks. Strong execution can reach main track but
is not guaranteed. This is "developable into a real paper," not a breakthrough.

**Must-cite / differentiate:** DBCD (AAAI'26, β-VAE counterfactual — heavy, generative), CADF /
CausalCDF (item-difficulty axis, not concept exposure), TSDR (arXiv:2605.05958 — propensity/DR but
KT, no mastery/DOA), **CMCD (NeurIPS'24 — closest: monotonic augmentation for fair CD; we differ by
per-concept exposure propensity + the stratified-DOA evaluation axis)**, CMES (AAAI'24), DisKCD
(untested-concept graph). **Citation hygiene:** do NOT cite "NCDLA" — it appears hallucinated; the
real noise-robust CD works are ORCDF (KDD'24) and DLLM (WWW'26).

## 1. What exists vs what must be built

The repo has **no CDM backbone**: `grep -niE 'doa|monoton|mastery|proficien|q_?matrix|ncdm|kancd'`
returns zero hits; `retricd/predictor.py` emits a single logit; `metrics.py` is AUC/ACC/RMSE/BCE.
So a real diagnostic core must be built. Everything below either REUSES or ADDS — nothing about the
data pipeline is thrown away.

| Reuse as-is | Extend | Add (new) | Demote to baseline |
|---|---|---|---|
| `data_retricd/` prepared splits | `metrics.py` (+DOA), `losses.py` (+IPS/DR), config/runner support, exposure counts | `excd/data.py`, `excd/model.py`, `excd/mnar_stress.py`, `excd/summarize.py` | RetriCD is retired; use Git history only for archaeology |

The current repo is a clean Ex-CD implementation. If a retrieval appendix is needed later, restore
RetriCD into a separate branch or separate baseline package instead of mixing it into this tree.

## 2. Phase 0 — Backbone + Figure-1 GATE  (≈1–1.5 wk)  ⟵ GO/NO-GO

**2.1 Q-matrix object** (`qmatrix.py`): build item×concept binary Q from the union of `cpt_seq`
per `exer_id` over the **train split only**. Persist a `[num_exercises, num_concepts]` sparse Q.
(Currently concepts live only per-response in `datasets.py` — there is no Q object.)

**2.2 NeuralCD/KaNCD head** (`cdm.py`):
- student proficiency `h_s ∈ (0,1)^K`, item difficulty `h_diff ∈ (0,1)^K`, discrimination scalar.
- interaction masked by Q row; **monotonicity** enforced via non-negative first-layer weights
  (NCDM) or a monotone MLP. KaNCD variant adds the low-rank latent-trait factorization.
- output per-(student,concept) **mastery** readout — this is the diagnostic artifact.

**2.3 DOA in `metrics.py`** (standard NeuralCD definition):
`DOA(k) = Σ_{a,b} δ(mast_{a,k}>mast_{b,k}) · Σ_j I_{jk} J_{ab,j} δ(r_{a,j}>r_{b,j}) /
          Σ_{a,b} δ(mast_{a,k}>mast_{b,k}) · Σ_j I_{jk} J_{ab,j}`,
where `I_{jk}=1` if item j tagged with concept k, `J_{ab,j}=1` if both a,b answered j. Report
mean over k + a **monotonicity-violation rate**. (Subsample student pairs for tractability on
junyi/nips34; fix the seed.)

**2.4 Exposure stratification** (`exposure.py`): per-concept train interaction count → decile
buckets. Implement **Exposure-Stratified DOA** = DOA(k) reported per concept-exposure decile +
the **head-minus-tail gap**. (Also support per-(student,concept) exposure as a secondary view.)

**2.5 GATE — `tools/analyze_exposure_cliff.py`:** on assist_09 + assist_17, plot DOA & monotonicity
per exposure decile for vanilla KaNCD.
- **PASS** (proceed): tail-decile DOA collapses toward ~0.5 while aggregate DOA stays high (~0.7).
- **FAIL** (stop & rethink): if the curve is flat, the failure mode is not real — do NOT proceed
  (this is the pre-validation that RetriCD skipped). Exposure skew is near-universal, so PASS is
  expected, but verify before committing weeks.

## 3. Phase 1 — Ex-CD method  (≈1 wk)

**3.1 Propensity** (`exposure.py`): per-concept observation propensity `p_c` from train frequency
(closed-form logistic/normalized-count; **no extra network**). For multi-concept items, aggregate
(mean) the concept propensities.

**3.2 IPS loss** (`losses.py`): `L_IPS = Σ w_i · BCE_i`, `w_i = 1/clip(p_{c(i)}, ε, 1)` (tune ε,
clip ceiling). Under-exposed concepts contribute proportionally more.

**3.3 Doubly-robust variant:** add a cheap imputation head reusing existing machinery;
`L_DR = imputation + IPS·(residual)`. Report IPS-only vs DR.

**3.4 Ablations:** vanilla CDM · IPS-only · DR · clip-level sweep · **propensity-misspecification
stress** (perturb the estimated propensity — document the circularity that propensity is estimated
from the same counts that define exposure, and bound overfitting-to-counts).

## 4. Phase 2 — MNAR-stress closed loop  (≈1 wk)  — repurposes your probes

The deletion/subset probes become the **controlled-skew harness**:
- **`mnar_stress.py`:** synthetically thin low-exposure concepts in train at increasing γ (gamma-
  style skew); chart Exposure-Stratified-DOA / monotonicity / AUC degradation for **Ex-CD vs
  vanilla vs IPS-only** as a function of γ.
- **corruption probe:** confirm Ex-CD's tail-DOA gains are not noise artifacts.
- **Negative control:** on dense, near-uniform-exposure sets (frcsub, math2) the head-tail gap
  should ≈ 0 and Ex-CD should match vanilla — proving the metric measures exposure, not sample size.
- **Pre-registered pass/fail:** the **tail-decile DOA gap must shrink** under Ex-CD vs vanilla,
  with bootstrap CIs (tail DOA is high-variance — report CIs, not point estimates). Aggregate
  AUC/DOA staying flat is acceptable and expected; the tail gap is the headline.

## 5. Experiment matrix

- **Primaries:** assist_09, assist_17, junyi, nips34 — chronological-per-student (realistic MNAR)
  **and** static-random (cross-check). Note: junyi has concept==exer_id (1:1) so its exposure
  story differs — treat as a stress case, not the headline.
- **Dense negative controls:** frcsub, math2 (near-uniform exposure → gap should vanish).
  (cdbd_a0910 is large/sparse — 17k items — NOT a dense control; use only for breadth.)
- **Breadth:** assist_12, assist_15, ednet_kt1 gap variants.
- **Metrics:** AUC/ACC/RMSE + aggregate DOA + **Exposure-Stratified DOA (per-decile + head-tail
  gap)** + monotonicity rate. DOA is PRIMARY (this is what makes it CD, not KT).
- **Baselines:** NeuralCD, KaNCD, RetriCD (retrieval baseline), IPS-only, DR; if time permits one
  heavy debias baseline (DBCD-/CMCD-style) to show existing debiasers also hide tail unreliability.

## 6. Risks & mitigations
- **Propensity circularity** (estimated from the counts that define exposure) → misspecification
  stress test + DR; document the bound. Treat as a known limitation, not a hidden one.
- **Tail-DOA statistical noise** → bootstrap/permutation CIs on the head-tail gap; subsample pairs
  with fixed seed.
- **CMCD collision** → frame the method as exposure-propensity (not sparsity-group augmentation)
  and anchor the contribution on the stratified-DOA evaluation axis.
- **"It's KT" reflex** → DOA + monotonicity as primary metrics; static-random cross-check; explicit
  diagnostic-artifact framing.

## 7. Sequencing
P0 (backbone+DOA+cliff gate) → **GATE** → P1 (IPS/DR) → P2 (MNAR-stress + controls) →
write-up. ~5–6 weeks solo. Do NOT start P1 until the Figure-1 cliff is reproduced.
