# Beyond Supervised Fine-Tuning

########################### WARNING ############################
When using the notebook there is one precaution to take

Due to pyarrow function the session will crash after few minutes, it's normal, just restart the Python runtime

########################### WARNING ############################

This repository supports a selected low-data annotation task from the configured task registry:

- `ds3_task6`: Congress tweet political superframe classification
- `ds1_task6`: content-moderation tweet superframe classification
- `ds4_task2`: news-article problem / solution / neutral classification
- label semantics and prompt text loaded from the selected task's mapping row
- Llama 3.1 8B or Gemma 3 12B with Unsloth, SFT, DPO, SFT->DPO, KTO, and SFT->KTO
- two SFT/DPO data-allocation variants:
  - `shared`
  - `sft25_dpo75`

The canonical study configs are:

- [llama31.yaml](configs/llama31.yaml)
- [gemma3.yaml](configs/gemma3.yaml)

## Data layout

The supported tasks use checked-in subset folders:

- `ds3_task6`: [data/raw_or_samples/ds_3__task_6](data/raw_or_samples/ds_3__task_6) and [dataset_task_mappings.csv](dataset_task_mappings.csv)
- `ds1_task6`: [data/raw_or_samples/ds_1__task_6](data/raw_or_samples/ds_1__task_6), including its task-local `dataset_task_mappings.csv`
- `ds4_task2`: [data/raw_or_samples/ds_4__task_2](data/raw_or_samples/ds_4__task_2), including its task-local `dataset_task_mappings.csv`

The pipeline:

- derives the available prompt columns, active prompt, canonical label set, and label column from the selected mapping row
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
- `notebooks/main_colab_train+infer_three_seeds.ipynb`: main Colab notebook

## Quickstart

Prepare the dataset:

```bash
python scripts/prepare_data.py --config configs/llama31.yaml --task ds1_task6
```

Run the baseline:

```bash
python scripts/score.py --config configs/llama31.yaml --task ds1_task6
```

Run one SFT pilot and build the alignment datasets:

```bash
python scripts/train.py sft --config configs/llama31.yaml --task ds1_task6 --seed 11 --train-size 100
python scripts/train.py tag-hard-cases --config configs/llama31.yaml --task ds1_task6 --seed 11 --train-size 100
python scripts/train.py build-preferences --config configs/llama31.yaml --task ds1_task6 --seed 11 --train-size 100
```

Run DPO and KTO:

```bash
python scripts/train.py dpo --config configs/llama31.yaml --task ds1_task6 --seed 11 --train-size 100 --preference-strategy hard_negative --start-from base
python scripts/train.py dpo --config configs/llama31.yaml --task ds1_task6 --seed 11 --train-size 100 --preference-strategy hard_negative --start-from sft
python scripts/train.py kto --config configs/llama31.yaml --task ds1_task6 --seed 11 --train-size 100 --start-from base
python scripts/train.py kto --config configs/llama31.yaml --task ds1_task6 --seed 11 --train-size 100 --start-from sft
```

Evaluate:

```bash
python scripts/aggregate.py --config configs/llama31.yaml --task ds1_task6
```

Run the full grid:

```bash
python scripts/train.py full --config configs/llama31.yaml --task ds1_task6 --include-random-wrong
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
- one desirable KTO row plus one undesirable KTO row for every incorrect label

The current KTO hyperparameters follow the conservative TRL recommendations for KTO rather than reusing the DPO learning rate:

- `learning_rate = 5e-6`
- `beta = 0.3`
- per-step batch size `4`
- effective batch size `16`
- `num_train_epochs = 2`
- `desirable_weight` is task-dependent: `3.0` for the 4-label tasks and `2.0` for the 3-label task
- `undesirable_weight = 1.0`

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
