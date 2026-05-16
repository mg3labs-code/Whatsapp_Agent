#!/bin/bash
# Run this locally with Railway DATABASE_URL to migrate production DB
# Usage: export DATABASE_URL="railway_url_here" && bash scripts/migrate_railway.sh
#
# Product catalog: load via Excel (not seed.py):
#   python -m scripts.import_product_price_list path/to/catalog.xlsx
echo "Running migrations on: $DATABASE_URL"
alembic upgrade head
echo "Migrations complete"
