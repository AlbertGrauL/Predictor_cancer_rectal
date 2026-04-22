import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import recall_score, precision_score, roc_auc_score, confusion_matrix

def calculate_metrics(y_true, y_probs, threshold=0.5):
    """
    Calcula métricas clínicas basadas en etiquetas reales y probabilidades.
    """
    y_true = y_true.cpu().numpy()
    y_probs = y_probs.cpu().numpy()
    y_pred = (y_probs >= threshold).astype(int)
    
    # Sensibilidad (Recall)
    sensitivity = recall_score(y_true, y_pred, zero_division=0)
    
    # Especificidad
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # AUC-ROC
    try:
        auc = roc_auc_score(y_true, y_probs)
    except ValueError:
        auc = 0.5 # Caso donde solo hay una clase en el batch
        
    return {
        "sensitivity": sensitivity,
        "specificity": specificity,
        "auc": auc,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": (tn, fp, fn, tp)
    }

def plot_confusion_matrix(y_true, y_probs, category_name, save_path, threshold=0.5):
    """
    Genera y guarda una imagen PNG de la matriz de confusión.
    """
    y_true = y_true.cpu().numpy()
    y_pred = (y_probs.cpu().numpy() >= threshold).astype(int)
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Sano', category_name.capitalize()],
                yticklabels=['Sano', category_name.capitalize()])
    plt.title(f'Matriz de Confusión: Especialista en {category_name.capitalize()}')
    plt.ylabel('Etiqueta Real')
    plt.xlabel('Predicción del Modelo')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
