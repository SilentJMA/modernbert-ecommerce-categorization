# Fine-tune ModernBERT for E-commerce Product Categorization

Train a text classifier that predicts top-level product categories from title, brand, and description.

## Overview

This project fine-tunes a transformer model for e-commerce taxonomy prediction.

- Base model: [`answerdotai/ModernBERT-base`](https://huggingface.co/answerdotai/ModernBERT-base)
- Dataset: [`Shopify/product-catalogue`](https://huggingface.co/datasets/Shopify/product-catalogue)
- Task: top-level product category classification
- Runtime: Google Colab (`T4 GPU` on free tier)

## Start Here

1. Open `modernbert_ecommerce_product_categorization.ipynb` in Google Colab.
2. Set runtime to `Python 3` + `T4 GPU`.
3. Run all cells in order.
4. Review `eval_accuracy`, `eval_f1_macro`, and `eval_f1_weighted`.

## Project File

- `modernbert_ecommerce_product_categorization.ipynb`: end-to-end workflow for preprocessing, training, evaluation, and inference.

## Notebook Workflow

1. Load dataset from Hugging Face.
2. Build training text from product title, brand, and description.
3. Convert taxonomy paths to top-level labels.
4. Tokenize text and encode labels.
5. Fine-tune `ModernBERT` for sequence classification.
6. Evaluate with accuracy and F1 metrics.
7. Generate class report + confusion matrix.
8. Run custom product predictions.
9. Save model artifacts.

## Training Setup (Macro-F1 Focused)

- `MAX_LENGTH = 256`
- `LEARNING_RATE = 1e-5`
- `NUM_EPOCHS = 5`
- `TRAIN_BATCH_SIZE = 16`
- `EVAL_BATCH_SIZE = 32`
- `WEIGHT_DECAY = 0.05`
- `WARMUP_RATIO = 0.1`
- `EARLY_STOPPING_PATIENCE = 2`
- Class-weighted cross-entropy loss
- Cosine learning-rate schedule

Sampling controls (for faster Colab iterations):
- `MAX_TRAIN_SAMPLES = 30000` (`None` for full train split)
- `MAX_EVAL_SAMPLES = 8000` (`None` for full eval split)

## Latest Evaluation Snapshot

Recent run on `8000` eval samples:
- `eval_loss: 1.0604`
- `eval_accuracy: 0.7318`
- `eval_f1_macro: 0.5506`
- `eval_f1_weighted: 0.7298`

Example predictions:
- `Apple iPhone 15 Pro Max...` -> `Electronics` (`0.997`)
- `Stainless Steel Non-Stick Frying Pan...` -> `Home & Garden` (`0.999`)
- `Organic Dry Cat Food...` -> `Animals & Pet Supplies` (`0.999`)

## Outputs

Trained artifacts are saved to:
- `./modernbert-ecommerce-topcat`

This folder includes model weights, config, and tokenizer files.

## Optional: Publish to Hugging Face Hub

1. Uncomment login and push lines in the last notebook cell.
2. Authenticate with your Hugging Face account.
3. Run that cell.

## Runtime Guidance

For Colab free tier:
- Use `T4 GPU`.
- Avoid CPU for full training runs.
- Avoid TPU unless you refactor for TPU/XLA.

If memory is limited, reduce `TRAIN_BATCH_SIZE` first.

## Troubleshooting

### `TrainingArguments.__init__() got an unexpected keyword argument 'evaluation_strategy'`
Use `eval_strategy="epoch"`.

### `Trainer.__init__() got an unexpected keyword argument 'tokenizer'`
Remove `tokenizer=tokenizer` from `Trainer(...)`.

### `RuntimeError: on_train_begin must be called before on_evaluate`
Remove `NotebookProgressCallback` and retry evaluation (already patched in notebook).

### `UndefinedMetricWarning` in sklearn report
Some rare classes may have no predicted samples in a run. This is expected under heavy class imbalance and mainly impacts macro metrics.

### Colab disconnects during training
1. Reduce `MAX_TRAIN_SAMPLES`.
2. Reduce `TRAIN_BATCH_SIZE`.
3. Reconnect and rerun from tokenization/training cells.
