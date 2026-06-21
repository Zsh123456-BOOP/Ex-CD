"""CertCD decision task: certification-efficient adaptive test design.

For each student, simulate administering items one at a time from the item pool and measure how
many items are needed to CERTIFY a target number of (structurally identifiable) concepts for that
student. We compare a certificate-greedy selection policy against standard test-design heuristics
(random, item popularity, max item-discrimination). A concept becomes certified once the student
has answered >= n_min informative items tagging it (and the concept is Q-structurally identifiable).

This shows the certificate has operational value for test design, not just post-hoc filtering.
It retrains the backbone to obtain item discriminations; it is a standalone experiment.

Example:
    python -m certcd.cat --dataset assist_09 --data-root data_retricd --output-dir outputs/cat --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from certcd.backbone import discrimination_by_exercise, train_backbone
from certcd.certificate import q_structure
from certcd.run import resolve_device
from excd.data import CDDataset, build_vocab, load_splits


def _student_target_concepts(test_ds, vocab, concept_ident, num_students):
    """Per student: the set of identifiable concepts that appear in their test items (what we'd
    want to diagnose for them)."""
    want = [set() for _ in range(num_students)]
    for s, cl in zip(test_ds.student, test_ds.concept_lists):
        for k in cl:
            if concept_ident[k] > 0:
                want[int(s)].add(int(k))
    return want


def simulate(student_targets, pool_items, item_concepts, item_pop, item_disc, concept_ident,
             policy: str, n_min: int, target_n: int, max_items: int, rng) -> float:
    """Return #items administered to certify target_n concepts for one student (or max_items)."""
    needed = set(student_targets)
    if len(needed) < target_n:
        return float("nan")  # student doesn't need that many; skip for a clean average
    cov = {}
    certified = set()
    # candidate items = those tagging at least one needed concept
    cand = [j for j in pool_items if any(k in needed for k in item_concepts[j])]
    if policy == "popularity":
        cand = sorted(cand, key=lambda j: -item_pop[j])
    elif policy == "max_disc":
        cand = sorted(cand, key=lambda j: -item_disc[j])
    elif policy == "random":
        rng.shuffle(cand)
    # certificate_greedy is dynamic (re-scored each step)
    administered = 0
    used = set()
    while administered < max_items and len(certified) < target_n:
        if policy == "certificate_greedy":
            best, best_gain = None, -1
            for j in cand:
                if j in used:
                    continue
                gain = sum(1 for k in item_concepts[j] if k in needed and k not in certified)
                if gain <= 0:
                    continue
                # tiebreak: prefer purer items (fewer extra concepts), then higher disc
                key_better = gain > best_gain
                if key_better:
                    best, best_gain = j, gain
            if best is None:
                break
            j = best
        else:
            j = next((x for x in cand if x not in used), None)
            if j is None:
                break
        used.add(j)
        administered += 1
        for k in item_concepts[j]:
            if k in needed:
                cov[k] = cov.get(k, 0) + 1
                if cov[k] >= n_min and concept_ident[k] > 0 and k not in certified:
                    certified.add(k)
    return float(administered) if len(certified) >= target_n else float(max_items)


def run_cat(args):
    device = resolve_device(args.device)
    train_df, valid_df, test_df = load_splits(args.dataset, args.data_root)
    vocab = build_vocab(train_df)
    train_ds = CDDataset(train_df, vocab, use_ips=False)
    valid_ds = CDDataset(valid_df, vocab, use_ips=False)
    test_ds = CDDataset(test_df, vocab, use_ips=False)

    model, _, _ = train_backbone(vocab, train_ds, valid_ds, model_type=args.model, epochs=args.epochs, device=device, seed=args.seed)
    item_disc = discrimination_by_exercise(model, device)  # [E]

    items_per_concept, has_pure, _, _ = q_structure(vocab)
    concept_ident = (has_pure & (items_per_concept >= 2)).astype(np.float32)

    # item pool = real exercises (skip UNK index 0); their concepts + popularity (train count)
    Q = vocab.q_matrix
    item_concepts = {j: np.where(Q[j] > 0)[0].tolist() for j in range(1, vocab.num_exercises)}
    pool_items = [j for j in range(1, vocab.num_exercises) if item_concepts[j]]
    item_pop = np.zeros(vocab.num_exercises, dtype=np.float64)
    np.add.at(item_pop, train_ds.exercise, 1.0)

    targets = _student_target_concepts(test_ds, vocab, concept_ident, vocab.num_students)
    eligible = [s for s in range(vocab.num_students) if len(targets[s]) >= args.target_n]
    rng = np.random.default_rng(args.seed)
    if len(eligible) > args.max_students:
        eligible = list(rng.choice(eligible, size=args.max_students, replace=False))

    policies = ["random", "popularity", "max_disc", "certificate_greedy"]
    result = {"dataset": args.dataset, "model": args.model, "target_n": args.target_n,
              "max_items": args.max_items, "n_min": args.n_min, "n_students": len(eligible),
              "items_to_target": {}}
    for pol in policies:
        vals = []
        for s in eligible:
            v = simulate(targets[s], pool_items, item_concepts, item_pop, item_disc, concept_ident,
                         pol, args.n_min, args.target_n, args.max_items, np.random.default_rng(args.seed + s))
            if not np.isnan(v):
                vals.append(v)
        result["items_to_target"][pol] = float(np.mean(vals)) if vals else float("nan")
        print(f"  policy {pol:<20} mean items-to-certify-{args.target_n} = {result['items_to_target'][pol]:.2f}", flush=True)

    os.makedirs(args.output_dir, exist_ok=True)
    path = os.path.join(args.output_dir, f"cat_{args.dataset}_{args.model}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  saved -> {path}", flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description="CertCD CAT certification-efficiency")
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--data-root", type=str, default="data_retricd")
    p.add_argument("--output-dir", type=str, default="outputs/cat")
    p.add_argument("--model", type=str, default="ncdm")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--target-n", type=int, default=8)
    p.add_argument("--max-items", type=int, default=120)
    p.add_argument("--n-min", type=int, default=2)
    p.add_argument("--max-students", type=int, default=400)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")
    run_cat(p.parse_args())


if __name__ == "__main__":
    main()
