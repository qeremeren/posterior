#!/usr/bin/env sh
# Download the Olist Brazilian e-commerce tables (CC BY-NC-SA 4.0, by Olist) from a public
# Hugging Face mirror of the Kaggle dataset, so no Kaggle account is needed.
set -e
DEST="${1:-data/olist}"
BASE="https://huggingface.co/datasets/bulutttt/olist-raw-data/resolve/main"
mkdir -p "$DEST"
for f in olist_customers_dataset olist_order_items_dataset olist_order_payments_dataset \
         olist_order_reviews_dataset olist_orders_dataset olist_products_dataset \
         olist_sellers_dataset product_category_name_translation; do
  curl -fsSL -o "$DEST/$f.csv" "$BASE/$f.csv"
  echo "$DEST/$f.csv"
done
