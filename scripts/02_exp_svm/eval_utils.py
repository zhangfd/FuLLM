import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score


def plot_confusion_matrices(y_true, y_pred, task_names, filename='confusion_matrices.png'):
    """Plot confusion matrices for multi-task classification"""
    n_tasks = len(task_names)
    fig, axes = plt.subplots(2, 3, figsize=(20, 15))
    axes = axes.ravel()

    accuracies = []

    for i, task_name in enumerate(task_names):
        # Ensure confusion matrix is 4x4
        cm = confusion_matrix(y_true[:, i], y_pred[:, i], labels=range(4))
        
        # Transpose confusion matrix
        cm = cm.T
        
        # Calculate accuracy
        accuracy = accuracy_score(y_true[:, i], y_pred[:, i])
        accuracies.append(accuracy)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], cbar=False)
        axes[i].set_title(f'{task_name}\nAccuracy: {accuracy:.2f}', fontsize=12)
        axes[i].set_xlabel('GT', fontsize=10)
        axes[i].set_ylabel('Predict', fontsize=10)
        axes[i].set_xticklabels(range(4))
        axes[i].set_yticklabels(range(4))

    # Remove extra subplots
    for j in range(n_tasks, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)

    return accuracies 