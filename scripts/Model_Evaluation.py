import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, precision_recall_curve, roc_auc_score, accuracy_score, auc, roc_curve

def evaluate_model(model, X_train, y_train, X_val, y_val, threshold=0.5, model_name="Model"):

    y_train_prob = model.predict_proba(X_train)[:, 1]
    y_val_prob = model.predict_proba(X_val)[:, 1]

    precision_val, recall_val, thresholds_pr = precision_recall_curve(y_val, y_val_prob)
    pr_auc_val = auc(recall_val, precision_val)

    fpr, tpr, thresholds_roc = roc_curve(y_val, y_val_prob)
    roc_auc_val = roc_auc_score(y_val, y_val_prob)

    if threshold == 'best':
        f1_scores = 2 * (precision_val * recall_val) / (precision_val + recall_val)
        best_threshold = thresholds_pr[np.argmax(f1_scores)]
        print(f"Best Threshold based on F1 score: {best_threshold:.4f}")
        threshold = best_threshold

    y_train_pred = (y_train_prob >= threshold).astype(int)
    y_val_pred = (y_val_prob >= threshold).astype(int)

    print(f"Training Set Classification Report (threshold = {threshold}):")
    print(classification_report(y_train, y_train_pred))

    print(f"Validation Set Classification Report (threshold = {threshold}):")
    print(classification_report(y_val, y_val_pred))

    print(f"ROC AUC (Validation): {roc_auc_val:.4f}")
    print(f"PR AUC (Validation): {pr_auc_val:.4f}")
    print(f"Accuracy (Validation): {accuracy_score(y_val, y_val_pred):.4f}")


    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc_val:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} - ROC Curve')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(recall_val, precision_val, color='red', label=f'PR Curve (AUC = {pr_auc_val:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{model_name} - Precision-Recall Curve')
    plt.legend()

    plt.tight_layout()
    plt.show()

    return threshold
