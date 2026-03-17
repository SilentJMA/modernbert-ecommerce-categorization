# ModernBERT Fine-Tuning for E-commerce Product Categorization

This project fine-tunes `answerdotai/ModernBERT-base` for e-commerce product categorization using the `Shopify/product-catalogue` dataset.

## Files
- `modernbert_ecommerce_product_categorization.ipynb`: End-to-end notebook for training, evaluation, and inference.

## What the notebook does
- Loads Shopify product catalog data from Hugging Face.
- Builds text inputs from product title, brand, and description.
- Converts hierarchical taxonomy labels to top-level categories.
- Fine-tunes `answerdotai/ModernBERT-base` for sequence classification.
- Evaluates with accuracy and F1 metrics.
- Runs custom prediction examples.
- Saves the fine-tuned model locally.

## Base Model Choice
The notebook uses `answerdotai/ModernBERT-base` because it is a strong encoder model for supervised text classification and is efficient to fine-tune for category prediction tasks.

## Quick start (Colab)
1. Open `modernbert_ecommerce_product_categorization.ipynb` in Google Colab.
2. Enable GPU runtime.
3. Run all cells in order.

## Optional
- Push trained artifacts to Hugging Face Hub by uncommenting the upload cell at the end of the notebook.
