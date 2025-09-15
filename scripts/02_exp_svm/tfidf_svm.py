import numpy as np
import jieba
import time
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report
from scipy.sparse import csr_matrix
from pathlib import Path

from data_utils import load_original_data, load_fold_data
from eval_utils import plot_confusion_matrices


def tokenize_chinese(text):
    """Chinese word segmentation"""
    return " ".join(jieba.cut(text))


def tfidf_svm_fold(fold_data_dir, output_prefix):
    """TF-IDF SVM with pre-split fold validation"""
    print(f"Running TF-IDF SVM with pre-split folds for {output_prefix}...")
    start_time = time.time()
    
    all_test_labels = []
    all_test_predictions = []
    all_test_dcaps = []
    task_names = ['随访信息来源', '受试者是否死亡', '受试者是否住院', '受试者是否手术', '受试者是否用药']
    
    for fold in range(1, 6):
        print(f"\nProcessing fold {fold}")
        
        # Load fold data
        (train_texts, train_labels, _), (test_texts, test_labels, test_dcaps) = load_fold_data(fold_data_dir, fold)
        
        # Text preprocessing
        train_tokenized = [tokenize_chinese(text) for text in train_texts]
        test_tokenized = [tokenize_chinese(text) for text in test_texts]
        
        # TF-IDF feature extraction
        vectorizer = TfidfVectorizer(max_features=5000)
        X_train_tfidf = vectorizer.fit_transform(train_tokenized)
        X_test_tfidf = vectorizer.transform(test_tokenized)
        
        X_train_tfidf = csr_matrix(X_train_tfidf).copy()
        X_test_tfidf = csr_matrix(X_test_tfidf).copy()
        
        # Training and prediction
        svm = SVC(kernel='rbf', probability=True)
        multi_target_svm = MultiOutputClassifier(svm, n_jobs=-1)
        multi_target_svm.fit(X_train_tfidf, train_labels)
        y_pred = multi_target_svm.predict(X_test_tfidf)
        
        all_test_labels.append(test_labels)
        all_test_predictions.append(y_pred)
        all_test_dcaps.extend(list(test_dcaps))
        
        # Single fold evaluation
        for i, task_name in enumerate(task_names):
            print(f"\n{task_name} Classification Report (Fold {fold}):")
            print(classification_report(test_labels[:, i], y_pred[:, i]))
    
    # Combine all fold results
    all_test_labels = np.vstack(all_test_labels)
    all_test_predictions = np.vstack(all_test_predictions)
    
    # Overall evaluation
    print("\nOverall Classification Report:")
    for i, task_name in enumerate(task_names):
        print(f"\n{task_name} Classification Report:")
        print(classification_report(all_test_labels[:, i], all_test_predictions[:, i]))
    
    # Save results
    results_df = pd.DataFrame()
    for i, task in enumerate(task_names):
        results_df[f"Pred_{task}"] = all_test_predictions[:, i]
        results_df[f"GT_{task}"] = all_test_labels[:, i]
    results_df['dCap'] = all_test_dcaps
    # mkdir if not exists
    output_xlsx_path = Path(f'data/output/{output_prefix}/{output_prefix}.xlsx')
    output_xlsx_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_excel(output_xlsx_path, index=False)
    plot_confusion_matrices(all_test_labels, all_test_predictions, 
                           ['follow_up_source', 'is_deceased', 'is_hospitalized', 'had_surgery', 'is_medicated'], 
                           f"data/output/{output_prefix}/{output_prefix}_confusion_matrix.png")
    
    end_time = time.time()
    print(f"\n总运行时间: {end_time - start_time:.2f} 秒")
    
    return all_test_labels, all_test_predictions


if __name__ == "__main__":
    # Run experiments
    tfidf_svm_fold("data/follow_up_train_data_aug", output_prefix="svm_tfidf_aug")
    tfidf_svm_fold("data/follow_up_train_data_wo_aug", output_prefix="svm_tfidf_wo_aug")
    print("TF-IDF SVM experiments completed") 