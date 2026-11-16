# Revision protocol

The configs `configs/llama31_revision.yaml` and `configs/gemma3_revision.yaml` extend the
original configs and write to separate `outputs/revision`, `results/revision` and
`data/processed_revision` folders. The original configs and commands behave as before.

## What changes

| Setting | Submitted manuscript | Revision protocol |
|---|---|---|
| Training seeds | 1 (3 identical inference passes) | 5 |
| Training subset | one fixed CSV per size | stratified draw per seed from the largest provided file |
| Post-SFT budget | epochs (KTO gets K times more steps) | same optimizer steps and batch size for every stage |
| Control arm | none | `sft_sft`: SFT continued with the same rows, steps and learning rate as SFT->DPO |
| Post-SFT learning rate | 5e-6 | grid 5e-6, 2e-5, 1e-4, all reported |
| DPO negatives | `random_wrong`, `hard_negative` | plus `hard_negative_oof` and `all_wrong` |
| Statistics | best configuration per cell | paired bootstrap vs `sft_sft` and vs `sft`, Holm-adjusted |

Notes:

- The budget is matched on optimizer steps and rows per step. One DPO row is a pair
  (two completions), so DPO still does two forward passes per row.
- DPO/KTO from the base model get the SFT stage's steps and batch size.
- The largest train size is the whole pool, so its seeds differ only in training randomness.
- `hard_negative_oof` trains K extra SFT models per cell (K = 5, fewer if a class is rarer).
- `summary.json` in each `preferences` folder reports how often each strategy picks the
  same rejected label as `hard_negative`.

## Commands

Run one cell first to check the environment:

```bash
python scripts/train.py protocol --config configs/llama31_revision.yaml --task ds4_task2 --seeds 11 --train-sizes 50 --learning-rates 2e-5
```

Full grid for one task (re-running skips finished stages):

```bash
python scripts/train.py protocol --config configs/llama31_revision.yaml --task ds4_task2
```

Statistics only (`results/revision/.../tables`):

```bash
python scripts/stats.py --config configs/llama31_revision.yaml --task ds4_task2
```

Generation versus log-prob scoring, on any run that already has `predictions_test.csv`:

```bash
python scripts/score_generation.py --task ds4_task2 --run baseline --mode greedy
```

```bash
python scripts/score_generation.py --task ds4_task2 --run seed_11/train_100/allocation_shared/sft --mode sample --num-samples 5 --temperature 1.0
```
