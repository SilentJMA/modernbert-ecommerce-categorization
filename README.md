# Fine-tune ModernBERT for e-commerce product categorization

Train a text classifier that predicts product categories from title, brand, and description.

This project uses:
- Base model: `answerdotai/ModernBERT-base`
- Dataset: `Shopify/product-catalogue`
- Task: top-level product taxonomy classification
- Environment: Google Colab (free tier works with T4 GPU)

## Start here

1. Open `modernbert_ecommerce_product_categorization.ipynb` in Google Colab.
2. Set runtime to `Python 3` + `T4 GPU`.
3. Run all cells in order.
4. Check `eval_accuracy`, `eval_f1_macro`, and `eval_f1_weighted`.

## Project file

- `modernbert_ecommerce_product_categorization.ipynb`: full workflow for data prep, training, evaluation, and inference.

## What the notebook does

1. Loads Shopify product data from Hugging Face.
2. Builds model input from `title + brand + description`.
3. Converts taxonomy paths to top-level categories.
4. Encodes labels and tokenizes text.
5. Fine-tunes `ModernBERT` for sequence classification.
6. Uses macro-F1-focused training settings.
7. Evaluates with accuracy/F1 + class report + confusion matrix.
8. Runs sample predictions and saves model artifacts.

## Current training defaults (macro-F1 oriented)

- `MAX_LENGTH = 256`
- `LEARNING_RATE = 1e-5`
- `NUM_EPOCHS = 5`
- `TRAIN_BATCH_SIZE = 16`
- `EVAL_BATCH_SIZE = 32`
- `WEIGHT_DECAY = 0.05`
- `WARMUP_RATIO = 0.1`
- `EARLY_STOPPING_PATIENCE = 2`
- Class-weighted cross-entropy loss for label imbalance
- Cosine learning-rate schedule

Sampling controls for faster Colab iteration:
- `MAX_TRAIN_SAMPLES = 30000` (set `None` for full train split)
- `MAX_EVAL_SAMPLES = 8000` (set `None` for full eval split)

## Latest evaluation snapshot

From your recent run (`8000` eval samples):
- `eval_loss: 1.0604`
- `eval_accuracy: 0.7318`
- `eval_f1_macro: 0.5506`
- `eval_f1_weighted: 0.7298`

Example predictions:
- `Apple iPhone 15 Pro Max...` -> `Electronics` (`0.997`)
- `Stainless Steel Non-Stick Frying Pan...` -> `Home & Garden` (`0.999`)
- `Organic Dry Cat Food...` -> `Animals & Pet Supplies` (`0.999`)

## Outputs

After training, artifacts are saved to:
- `./modernbert-ecommerce-topcat`

This folder contains model weights, config, and tokenizer files.

## Optional publishing

To push model artifacts to Hugging Face Hub:
1. Uncomment login/push lines in the last notebook cell.
2. Authenticate with your Hugging Face account.
3. Run that cell.

## Runtime guidance

For Colab free tier:
- Use `T4 GPU`.
- Avoid CPU for full training runs.
- Avoid TPU unless you refactor for TPU/XLA.

If memory is tight, lower `TRAIN_BATCH_SIZE` first.

## Troubleshooting

### `TrainingArguments.__init__() got an unexpected keyword argument 'evaluation_strategy'`
Use `eval_strategy="epoch"`.

### `Trainer.__init__() got an unexpected keyword argument 'tokenizer'`
Remove `tokenizer=tokenizer` from `Trainer(...)`.

### `RuntimeError: on_train_begin must be called before on_evaluate`
Notebook callback state issue. Remove `NotebookProgressCallback` and retry evaluate (already patched in the notebook).

### `UndefinedMetricWarning` from sklearn classification report
Some rare classes have no predicted samples in that run. This is expected with extreme class imbalance. It mainly affects macro metrics.

### Colab disconnects during training
1. Reduce `MAX_TRAIN_SAMPLES`.
2. Reduce `TRAIN_BATCH_SIZE`.
3. Reconnect and rerun from tokenization/training cells.
