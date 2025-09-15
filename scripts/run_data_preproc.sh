#!/bin/bash
echo "Running data_proc.sh..."

echo "Running data_rewrite.py..."
python scripts/01_data_preproc/data_rewrite.py

echo "Running data_synthesis.py..."
python scripts/01_data_preproc/data_synthesis.py

echo "Running data_filter.py..."
python scripts/01_data_preproc/data_filter.py

echo "Running split_folds.py..."
python scripts/01_data_preproc/split_folds.py

echo "Running prepare_train_pred.py..."
python scripts/01_data_preproc/prepare_train_pred.py