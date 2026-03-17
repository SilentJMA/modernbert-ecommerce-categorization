# Fine-tune ModernBERT for e-commerce product categorization

Train a text classifier that predicts product categories from title, brand, and description.

This project uses:
- Base model: `answerdotai/ModernBERT-base`
- Dataset: `Shopify/product-catalogue`
- Task: top-level product taxonomy classification
- Environment: Google Colab (free tier works with T4 GPU)

## Start here

If you want the fastest path:
1. Open `modernbert_ecommerce_product_categorization.ipynb` in Google Colab.
2. Set runtime to `Python 3` + `T4 GPU`.
3. Run all cells in order.
4. Review `accuracy`, `f1_macro`, and `f1_weighted` in the evaluation cells.

## What problem this solves

Many product catalogs have inconsistent category labels. This notebook gives you a repeatable way to:
- Standardize category assignment.
- Reduce manual categorization time.
- Produce a model you can reuse for bulk or real-time labeling.

## Why this model

`answerdotai/ModernBERT-base` is a strong fit because:
- It is an encoder model, which is ideal for supervised classification.
- It supports longer context than older BERT-style models.
- It is efficient enough to fine-tune on limited GPU resources.

## Project file

- `modernbert_ecommerce_product_categorization.ipynb`: complete workflow for setup, training, evaluation, and prediction.

## How the notebook works

The notebook follows this flow:
1. Install dependencies.
2. Load Shopify dataset from Hugging Face.
3. Build one text field from product title, brand, and description.
4. Convert hierarchical category paths to top-level labels.
5. Tokenize text with `AutoTokenizer`.
6. Fine-tune `AutoModelForSequenceClassification`.
7. Evaluate with accuracy and F1.
8. Run sample predictions.
9. Save model artifacts locally.

## Runtime and compute guidance

For Google Colab free tier:
- Choose `T4 GPU`.
- Do not use CPU for full training unless you only test a very small sample.
- Do not use TPU unless you plan to refactor for TPU/XLA.

Current default config in notebook:
- `MAX_LENGTH = 256`
- `NUM_EPOCHS = 3`
- `TRAIN_BATCH_SIZE = 16`
- `EVAL_BATCH_SIZE = 32`
- `MAX_TRAIN_SAMPLES = 30000` (set to `None` for full train split)
- `MAX_EVAL_SAMPLES = 8000` (set to `None` for full test split)

If Colab runs out of memory, lower `TRAIN_BATCH_SIZE` first.

## Outputs

After training, artifacts are saved to:
- `./modernbert-ecommerce-topcat`

Saved files include model weights, config, and tokenizer files needed for reuse.

## Optional publishing

You can publish the fine-tuned model to Hugging Face Hub:
1. Uncomment the login and `push_to_hub` lines in the last notebook cell.
2. Authenticate with your Hugging Face account.
3. Run the upload cell.

## Troubleshooting

### Error: `TrainingArguments.__init__() got an unexpected keyword argument 'evaluation_strategy'`
Use `eval_strategy="epoch"` instead of `evaluation_strategy="epoch"`.

### Error: `Trainer.__init__() got an unexpected keyword argument 'tokenizer'`
Remove the `tokenizer=tokenizer` argument from `Trainer(...)`.

### Runtime disconnects during training
Try these changes:
1. Reduce `MAX_TRAIN_SAMPLES`.
2. Reduce `TRAIN_BATCH_SIZE`.
3. Reconnect and restart from the tokenization/training section.

## Documentation style notes

This README is written to follow Digital.gov documentation guidance:
- Plain language and short sentences.
- Task-first headings and clear action steps.
- Scannable structure with concise lists.

References:
- [Digital.gov style guide](https://digital.gov/style-guide)
- [PlainLanguage.gov guidelines](https://www.plainlanguage.gov/guidelines/)

## Repository

- GitHub: [modernbert-ecommerce-categorization](https://github.com/SilentJMA/modernbert-ecommerce-categorization)
