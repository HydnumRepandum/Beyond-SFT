# Beyond Supervised Fine-Tuning

This repository now targets one task only:

- `ds_3 / task_6` political tweet superframe classification
- four labels: `A`, `B`, `C`, `D`
- label semantics loaded from `dataset_task_mappings.csv`
- Llama 3.1 8B or Gemma 3 12B with Unsloth, SFT, DPO, SFT->DPO, KTO, and SFT->KTO
- two SFT/DPO data-allocation variants:
  - `shared`
  - `sft25_dpo75`

The canonical study configs are:

- [llama31.yaml](configs/llama31.yaml)
- [gemma3.yaml](configs/gemma3.yaml)

## Data layout

The repo now uses the provided subset files directly:

- [dataset_task_mappings.csv](dataset_task_mappings.csv)
- [ds_3__task_6_train_set_50.csv](data/raw_or_samples/ds_3__task_6_train_set_50.csv)
- [ds_3__task_6_train_set_100.csv](data/raw_or_samples/ds_3__task_6_train_set_100.csv)
- [ds_3__task_6_train_set_250.csv](data/raw_or_samples/ds_3__task_6_train_set_250.csv)
- [ds_3__task_6_train_set_500.csv](data/raw_or_samples/ds_3__task_6_train_set_500.csv)
- [ds_3__task_6_eval_set.csv](data/raw_or_samples/ds_3__task_6_eval_set.csv), treated as the held-out test file

The pipeline:

- derives the prompt and canonical label set from the mapping row for dataset `3`, task `6`
- uses the provided train subsets as-is

## Repo structure

- `configs/llama31.yaml`: Llama 3.1 8B study configuration
- `configs/gemma3.yaml`: Gemma 3 12B study configuration
- `data/raw_or_samples/`: checked-in raw/sample CSV inputs
- `data/processed/`: generated preparation artifacts, ignored by Git
- `scripts/prepare_data.py`: prepare processed train/test splits and train-size subsets
- `scripts/score.py`: pretrained baseline inference
- `scripts/train.py`: SFT, hard-case tagging, preference construction, DPO/KTO, and full-grid orchestration
- `scripts/aggregate.py`: aggregated tables and plots
- `notebooks/colab_topic_alignment.ipynb`: main Colab notebook
- `notebooks/experimental/colab_train_once_infer_three_seeds.ipynb`: experimental notebook; keep outputs as exploratory only

## Quickstart

Prepare the dataset:

```bash
python scripts/prepare_data.py --config configs/llama31.yaml
```

Run the baseline:

```bash
python scripts/score.py --config configs/llama31.yaml
```

Run one SFT pilot and build the alignment datasets:

```bash
python scripts/train.py sft --config configs/llama31.yaml --seed 11 --train-size 100
python scripts/train.py tag-hard-cases --config configs/llama31.yaml --seed 11 --train-size 100
python scripts/train.py build-preferences --config configs/llama31.yaml --seed 11 --train-size 100
```

Run DPO and KTO:

```bash
python scripts/train.py dpo --config configs/llama31.yaml --seed 11 --train-size 100 --preference-strategy hard_negative --start-from base
python scripts/train.py dpo --config configs/llama31.yaml --seed 11 --train-size 100 --preference-strategy hard_negative --start-from sft
python scripts/train.py kto --config configs/llama31.yaml --seed 11 --train-size 100 --start-from base
python scripts/train.py kto --config configs/llama31.yaml --seed 11 --train-size 100 --start-from sft
```

Evaluate:

```bash
python scripts/aggregate.py --config configs/llama31.yaml
```

Run the full grid:

```bash
python scripts/train.py full --config configs/llama31.yaml --include-random-wrong
```

## DPO and KTO dataset construction

Each alignment example starts from the same fixed prompt and the same gold JSON answer:

- `chosen`: the gold JSON label, for example `{"label":"B"}`
- `rejected`: one incorrect JSON label under the same prompt

Two rejected-label strategies are implemented:

1. `random_wrong`
- sample one of the other labels uniformly

2. `hard_negative`
- score all candidate JSON labels with the SFT pilot
- choose the highest-scoring wrong label

This makes DPO a strict preference task over alternative closed-label responses instead of free-form completions.

KTO is built as an all-label binary-desirability dataset:

- one desirable row: `prompt` + gold completion + `label = True`
- one undesirable row for each wrong label: `prompt` + wrong completion + `label = False`

So each original training example yields:

- one paired DPO row
- four unpaired KTO rows in this 4-class task

The current KTO hyperparameters follow the conservative TRL recommendations for KTO rather than reusing the DPO learning rate:

- `learning_rate = 5e-6`
- `beta = 0.3`
- per-step batch size `4`
- effective batch size `16`
- `num_train_epochs = 2`
- `desirable_weight = 3.0` and `undesirable_weight = 1.0` so the single positive row is not underweighted against the three negative rows

Each trained adapter directory now also includes `model_metadata.json`, documenting:

- base model IDs
- adapter run path and run metadata
- quantization mode
- prompt format
- a stable hash of the study config

The default full pipeline also evaluates two stage-allocation regimes:

- `shared`: SFT and DPO both use the full selected train subset
- `sft25_dpo75`: SFT trains on a stratified 25% split of the selected train subset, and DPO/KTO preferences are built from the remaining 75%

Base-started DPO/KTO are reported only for `shared`; the 25/75 allocation reports the SFT-started post-training variants.
