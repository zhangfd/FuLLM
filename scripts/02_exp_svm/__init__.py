"""
SVM Experiments Package

Contains two SVM classification algorithms:
1. TF-IDF SVM
2. Word2Vec SVM

Each algorithm supports two modes:
- Original data with 5-fold cross validation (without augmentation)
- Pre-split fold data validation (with augmentation)
"""

from .tfidf_svm import tfidf_svm_cv, tfidf_svm_fold
from .word2vec_svm import word2vec_svm_cv, word2vec_svm_fold

__version__ = "1.0.0"
