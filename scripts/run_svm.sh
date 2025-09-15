#!/bin/bash
echo "Running svm.sh..."

echo "Running tfidf_svm.py..."
python scripts/02_exp_svm/tfidf_svm.py

echo "Running word2vec_svm.py..."
python scripts/02_exp_svm/word2vec_svm.py