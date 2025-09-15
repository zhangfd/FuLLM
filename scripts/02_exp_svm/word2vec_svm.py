import numpy as np
import jieba
import time
import pandas as pd
from gensim.models import Word2Vec
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report
from pathlib import Path

from data_utils import load_original_data, load_fold_data
from eval_utils import plot_confusion_matrices


def tokenize_chinese(text):
    """Chinese word segmentation"""
    return list(jieba.cut(text))


def document_vector(doc, w2v_model):
    """Create document vector by averaging word vectors"""
    vectors = [w2v_model.wv[word] for word in doc if word in w2v_model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(w2v_model.vector_size)


def word2vec_svm_fold(fold_data_dir, output_prefix):
    """Word2Vec SVM with pre-split fold validation"""
    print(f"Running Word2Vec SVM with pre-split folds for {output_prefix}...")
    start_time = time.time()
    
    # Collect all training data for Word2Vec model training
    all_train_texts = []
    for fold in range(1, 6):
        (train_texts, _, _), _ = load_fold_data(fold_data_dir, fold)
        all_train_texts.extend(train_texts)
    
    # Train Word2Vec model
    print("Training Word2Vec model on all training data...")
    tokenized_texts = [tokenize_chinese(text) for text in all_train_texts]
    w2v_model = Word2Vec(sentences=tokenized_texts, vector_size=100, window=5, min_count=1, workers=4)
    
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
        
        # Create document vectors
        train_vectors = np.array([document_vector(doc, w2v_model) for doc in train_tokenized])
        test_vectors = np.array([document_vector(doc, w2v_model) for doc in test_tokenized])
        
        # Training and prediction
        svm = SVC(kernel='rbf', probability=True)
        multi_target_svm = MultiOutputClassifier(svm, n_jobs=-1)
        multi_target_svm.fit(train_vectors, train_labels)
        y_pred = multi_target_svm.predict(test_vectors)
        
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
    word2vec_svm_fold("data/follow_up_train_data_aug", output_prefix="svm_w2v_aug")
    word2vec_svm_fold("data/follow_up_train_data_wo_aug", output_prefix="svm_w2v_wo_aug")
    print("Word2Vec SVM experiments completed")