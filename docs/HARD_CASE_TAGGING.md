# Hard-Case Tagging Explained

This note explains how the pipeline constructs test hard-case tags and how the `hard_negative` DPO pairs are constructed from SFT pilot scores.

## Why this exists

In a classification task, not all examples are equally difficult.

Some examples are:

- easy, because the model strongly prefers one label
- ambiguous, because two labels look similarly plausible
- genuinely wrong for the current model, because the top prediction is incorrect

The pipeline records that difficulty information after SFT for diagnostics and reporting. DPO pair construction uses the SFT pilot scores directly, but it no longer depends on the tagging script.

## What the SFT pilot produces

After SFT training, the model scores every candidate label completion for each example.

For a 4-class task, the model compares four possible answers under the same prompt:

- `{"label":"A"}`
- `{"label":"B"}`
- `{"label":"C"}`
- `{"label":"D"}`

For each example, the pipeline stores:

- the score of each label candidate
- the predicted label
- whether that prediction is correct
- the gap between the best and second-best label
- the highest-scoring wrong label, called `hard_negative_label`

That scoring logic is implemented in [inference.py](../src/topic_alignment/inference.py).

## The two hard-case tags

The pipeline defines two kinds of hard cases for test-set aggregation.

### `is_hard_primary`

This is the low-margin definition.

The model sorts the candidate labels from best to worst. Then it computes:

`score_margin = score(best_label) - score(second_best_label)`

If that margin is small, the example is considered difficult because the model is not clearly separating the best answer from the runner-up.

The threshold is set from the SFT train split:

- method: use the margin distribution on `predictions_train.csv`
- cutoff: the configured quantile, currently `0.25`

The train-derived cutoff is then applied to the test predictions. So `is_hard_primary = True` means:

- this test example falls below the train-calibrated low-margin cutoff
- the model is uncertain here, even if it may still be correct

### `is_hard_secondary`

This is the misclassification definition.

It is much simpler:

- `True` if the SFT pilot prediction is wrong
- `False` if the SFT pilot prediction is correct

So `is_hard_secondary = True` means:

- the current SFT pilot actually failed on this example

## Example

Suppose the gold label is `C`, and the SFT pilot scores:

- `A`: `-1.80`
- `B`: `-1.25`
- `C`: `-1.10`
- `D`: `-2.05`

Then:

- predicted label = `C`
- second-best label = `B`
- `score_margin = -1.10 - (-1.25) = 0.15`
- `hard_negative_label = B`

Interpretation:

- the model is correct
- but `B` is close behind
- `B` is the wrong answer the model currently finds most plausible
- if `0.15` is below the train-derived cutoff, then `is_hard_primary = True`
- because the model is still correct, `is_hard_secondary = False`

Now suppose the scores are:

- `A`: `-1.95`
- `B`: `-1.05`
- `C`: `-1.30`
- `D`: `-2.10`

Then:

- predicted label = `B`
- gold label = `C`
- `hard_negative_label = B`
- `is_hard_secondary = True`

This example is hard in the stronger sense: the model is actually wrong.

## What `hard_negative_label` means

The `hard_negative_label` is not the same thing as `is_hard_primary` or `is_hard_secondary`.

It is simply:

- the highest-scoring incorrect label for that example

In other words, it is the wrong answer the model currently finds most plausible.

This is exactly the label used for `hard_negative` DPO construction.

## How DPO uses it

For DPO, each preference row keeps the prompt fixed and compares:

- `chosen`: the gold JSON label
- `rejected`: one wrong JSON label

Under the `hard_negative` strategy, the rejected answer is:

- the `hard_negative_label`

So the DPO pair says:

- "Under this prompt, prefer the correct label over the most plausible wrong label."

This creates a materially harder comparison than pairing the gold label against a wrong label the model already rejects.

## Important distinction

Hard-case tagging and hard-negative pair building are related, but they are not identical.

Hard-case tagging uses:

- `score_margin`
- correctness
- a train-derived margin threshold applied to test predictions

Hard-negative DPO pair building uses:

- the stored `hard_negative_label`

That means:

- the tagging script is not needed to construct DPO pairs
- `is_hard_primary` is not written onto training or preference-source predictions
- DPO construction only needs the SFT prediction files, because they already contain `hard_negative_label`

The tagging step is still useful for diagnostics and for reporting that wants to separate easy and hard test examples.

## Where this happens in the pipeline

Relevant files:

- [inference.py](../src/topic_alignment/inference.py)
  This computes per-label scores, `score_margin`, predicted label, and `hard_negative_label`.

- [train.py](../scripts/train.py) subcommand `tag-hard-cases`
  This computes the train-derived low-margin threshold and writes tagged test predictions.

- [train.py](../scripts/train.py) subcommand `build-preferences`
  This uses `hard_negative_label` to build the `hard_negative` DPO dataset.

## Takeaway

The simplest way to think about it is:

- `is_hard_primary`: the model is uncertain on a test example
- `is_hard_secondary`: the model is wrong on a test example
- `hard_negative_label`: the wrong answer the model likes most

For post-training, the `hard_negative_label` is the key object because it tells DPO which plausible incorrect answer should be compared against the correct one.
