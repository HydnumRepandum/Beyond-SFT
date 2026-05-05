# Methodological Decisions

## Scope of the study

The study is intentionally narrow but now task-selectable:

- three supported task keys: `ds3_task6`, `ds1_task6`, and `ds4_task2`
- two configured base model families: Llama 3.1 8B Instruct and Gemma 3 12B
- one active prompt source per run, selected by `format.prompt_source`
- one test protocol
- one PEFT setup
- one SFT stage and two post-training families: DPO and KTO

The selected task is resolved from the config `task_registry` before data processing, training, scoring, or aggregation starts.

## Task metadata, prompt, and labels

Task metadata comes from the selected task's mapping row. For the checked-in default task this is [dataset_task_mappings.csv](../dataset_task_mappings.csv); for `ds1_task6` and `ds4_task2` it is `/Users/ematheve/Desktop/dataset task RL/dataset_task_mappings.csv`.

The selected task defines:

The pipeline derives:

- all available prompt columns from the mapping row
- the active prompt from `format.prompt_source`
- the canonical output key: `label`
- the response prefix: `\nAnswer:\n`
- the canonical label IDs
- the raw label column

For `ds4_task2`, the mapping row calls the third label `NEUTRAL`, while the provided CSV uses `Neither`; the config records that alias so both values map to label `C`.

The model is trained and evaluated as a closed-label JSON generation task. A canonical completion is a short JSON object such as `{"label":"C"}`.

## Data sources and split design

Each task uses provided subset files directly:

- `ds3_task6`: checked-in files under [data/raw_or_samples](../data/raw_or_samples), train sizes `50`, `100`, `250`, `500`
- `ds1_task6`: external files under `/Users/ematheve/Desktop/dataset task RL/ds_1__task_6__training_data_superframes_final`, train sizes `50`, `100`, `250`, `500`
- `ds4_task2`: external files under `/Users/ematheve/Desktop/dataset task RL/ds_4__task_2__training_data_newsarticles_problem_solution_final`, train sizes `50`, `100`, `250`, `500`, `1000`

The current pipeline does not create any intermediate split. It keeps the provided held-out file intact as one `test` split.

The active split settings are:

- `split_seed = 1729`
- subset seeds for repeated runs: `11`, `22`, `33`
- requested and provided train sizes are task-specific
- `full_train_source` and `reference_train_size` are task-specific

The configured `reference_train_size` is used for full-train reporting tables and figures.

The run seeds capture initialization, adapter allocation, trainer shuffling, and seeded preference-choice variance. They are not independent data-resampling seeds, because the train subsets are fixed CSV files reused across runs.

## Base model and inference settings

The default Llama study configuration is:

- base family: `Llama 3.1 8B Instruct`
- `base_model_id = unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit`
- `max_seq_length = 4096`
- `load_in_4bit = true`

Inference settings used for scoring are:

- classification is deterministic
- the pipeline does not sample free-form tokens at inference time
- the score for each label is the mean token log-probability of its canonical JSON completion
- prediction is the highest-scoring label among the closed candidate set
- `save_logprob_predictions = true`

Precision is selected dynamically from hardware capability:

- `bf16 = true` when GPU capability supports it
- `fp16 = false` in all cases
- if bf16 is unavailable, the trainers fall back to non-fp16 execution rather than enabling fp16 automatically

Common trainer/runtime settings across the trainable stages:

- the run seed is the experiment seed for that condition
- `report_to = ["comet_ml"]` when `tracking.enabled = true` and `COMET_API_KEY` is present in the environment
- `report_to = []` otherwise
- `remove_unused_columns = false`
- `dataset_num_proc = 1` where the trainer configuration exposes it

## Experiment tracking

The current pipeline uses Comet ML for trainer-level experiment tracking.

Tracking configuration:

- `tracking.enabled = true`
- `tracking.provider = comet_ml`
- `tracking.project_name = beyond_sft_superframe_alignment_ds3_task6`
- `tracking.mode = create`
- `tracking.start_online = true`
- `tracking.log_assets = true`

In Colab, the API key is provided through the `COMET_API_KEY` secret and exposed as an environment variable before training subprocesses start. If `COMET_API_KEY` is absent, the code disables external trainer reporting and runs only with local file artifacts.

The Comet integration is attached through Hugging Face trainer callbacks for:

- SFT
- DPO
- KTO

Each trainer run uses a deterministic `run_name` derived from the saved run directory, so the Comet experiment name is stable across reruns of the same condition. Baseline inference and the final aggregation scripts remain file-based and are not currently logged as separate Comet experiments.

## PEFT / LoRA setup

All trainable stages use LoRA adapters with the following settings:

- `r = 16`
- `lora_alpha = 32`
- `lora_dropout = 0.05`
- `bias = none`
- `gradient_checkpointing = unsloth`

Target modules:

- `q_proj`
- `k_proj`
- `v_proj`
- `o_proj`
- `gate_proj`
- `up_proj`
- `down_proj`

The pipeline saves adapter-only checkpoints plus metadata, not merged full-model checkpoints by default.

## SFT design and hyperparameters

SFT is the first trainable stage. It is run once for every:

- seed in `11, 22, 33`
- train size in `50, 100, 250, 500`
- data allocation variant in `shared`, `sft25_dpo75`

SFT trains on the selected train subset only. The held-out `test` split is not passed to the trainer and is used only for post-training scoring.

SFT hyperparameters:

- `num_train_epochs = 3`
- `per_device_train_batch_size = 4`
- `gradient_accumulation_steps = 2`
- effective train batch size: `8`
- `learning_rate = 2e-4`
- `warmup_ratio = 0.05`
- `weight_decay = 0.01`
- `logging_steps = 5`
- `evaluation_strategy = no`
- `save_strategy = epoch`
- `save_total_limit = 2`
- `optim = adamw_8bit`
- `packing = false`

After SFT training, the pipeline scores:

- the SFT train rows
- the preference-source rows
- the full test split

Those prediction files are later reused to construct DPO and KTO datasets.

## Data-allocation variants

The study evaluates two allocation variants for post-training:

### `shared`

- SFT trains on the full selected train subset
- preference construction uses that same full subset

### `sft25_dpo75`

- SFT trains on a deterministic stratified 25% split of the selected train subset
- DPO and KTO datasets are built from the remaining 75%
- `split_allocation_sft_fraction = 0.25`

This is treated as a separate reported condition rather than a replacement for the shared-data setting.

## Hard-case tagging and difficulty definition

Hard-case tagging is still used internally for:

- diagnostic tagging

Primary difficulty definition:

- method: `low_margin`
- statistic: top-1 minus top-2 candidate score margin
- threshold source: the SFT `train` predictions
- threshold quantile: `0.25`

Secondary difficulty definition:

- method: `pilot_misclassified`
- criterion: the SFT pilot prediction is wrong

These tags are not currently reported as separate published result subsets; the reported tables use `all_examples`.

## DPO dataset construction

DPO uses paired preference data. For each preference-source example:

- the prompt is fixed
- the `chosen` completion is the gold JSON label
- the `rejected` completion is one wrong JSON label

The pipeline supports two rejected-label strategies:

### `random_wrong`

- sample uniformly from the incorrect labels
- sampling is seeded by the run seed

### `hard_negative`

- use the SFT pilot trained on the same train subset
- score all canonical label completions
- choose the highest-scoring incorrect label
- if the stored hard negative is invalid, fall back to the highest-scoring remaining wrong label
- this gives DPO a real correction signal by comparing the gold response against the wrong response the SFT pilot currently finds most plausible

Each source example contributes:

- one DPO row

## DPO training design and hyperparameters

DPO is trained in two initialization modes:

### `base -> DPO`

- policy model: base model + fresh LoRA adapter
- reference model: frozen base model
- reported only for the `shared` allocation, where the whole selected train subset is available for preference construction

### `SFT -> DPO`

- policy model: SFT adapter loaded as trainable
- reference model: the same SFT adapter loaded as frozen
- reported for both `shared` and `sft25_dpo75`

DPO hyperparameters:

- `num_train_epochs = 2`
- `per_device_train_batch_size = 4`
- `gradient_accumulation_steps = 4`
- effective train batch size: `16`
- `learning_rate = 5e-6`
- `warmup_ratio = 0.05`
- `weight_decay = 0.0`
- `beta = 0.5`
- `logging_steps = 5`
- `evaluation_strategy = no`
- `save_strategy = epoch`
- `save_total_limit = 2`
- `optim = adamw_8bit`
- `max_prompt_length = 1024`
- `max_length = 1152`
- `loss_type = sigmoid`

Important implementation detail:

- trainer-side evaluation is disabled because `evaluation_strategy = no` and `eval_dataset = None`
- DPO test scoring is performed after training by scoring the full `test` split with the resulting adapter

The primary reported DPO strategy in plots is `hard_negative`, but the experiment grid can also include `random_wrong`.

## KTO dataset construction

KTO uses unpaired desirability rows rather than DPO-style chosen/rejected pairs.

For each source example in this 4-class task, the pipeline creates:

- one desirable row with the gold JSON completion and `label = True`
- one undesirable row for each incorrect label completion and `label = False`

So each original example contributes:

- one positive KTO row
- three negative KTO rows
- four KTO rows total

The current KTO dataset strategy is:

- `dataset_strategy = all_wrong`

This means KTO does not use `random_wrong` versus `hard_negative` as separate strategies. It always trains against all wrong labels.

## KTO training design and hyperparameters

KTO is also trained in two initialization modes:

### `base -> KTO`

- policy model: base model + fresh LoRA adapter
- reference model: frozen base model
- reported only for the `shared` allocation

### `SFT -> KTO`

- policy model: SFT adapter loaded as trainable
- reference model: the same SFT adapter loaded as frozen
- reported for both `shared` and `sft25_dpo75`

KTO hyperparameters:

- `dataset_strategy = all_wrong`
- `num_train_epochs = 2`
- `per_device_train_batch_size = 4`
- `gradient_accumulation_steps = 4`
- effective train batch size: `16`
- `learning_rate = 5e-6`
- `warmup_ratio = 0.05`
- `weight_decay = 0.0`
- `beta = 0.3`
- `logging_steps = 5`
- `evaluation_strategy = no`
- `save_strategy = epoch`
- `save_total_limit = 2`
- `max_prompt_length = 1024`
- `max_length = 1152`
- `loss_type = kto`
- `desirable_weight = 3.0`
- `undesirable_weight = 1.0`
- `disable_dropout = true`
- `optim = adamw_8bit`

Important implementation details:

- KTO trainer-side evaluation is disabled because `evaluation_strategy = no` and `eval_dataset = None`
- KTO test scoring is performed after training by scoring the full `test` split
- the `3:1` positive/negative weighting is chosen because each prompt yields one desirable row and three undesirable rows

## Test reporting

The current reporting layer evaluates on one subset only:

- `all_examples`

Metrics:

- accuracy
- macro F1
- per-class F1
- confusion matrices

The summary table reports:

- mean across seeds
- standard deviation across seeds

The current results table does not report standard error; the plotted error bars are standard deviations.

## Reproducibility artifacts

For each completed run, the pipeline writes:

- run metadata
- LoRA adapter checkpoints
- `model_metadata.json`
- trainer log history
- training metrics
- prediction files
- preference datasets
- aggregated result tables and figures

`model_metadata.json` documents:

- base model family and model IDs
- adapter path and run metadata
- quantization mode
- prompt format
- config hash

## Practical note on runtime

The pipeline is optimized for Colab-style GPU use:

- Unsloth 4-bit loading is used for the base model
- adapters are trained instead of full-model fine-tuning
- the full grid can still require multiple Colab sessions
- the notebook resume logic skips already completed artifact-producing stages when rerun
