"""
Module de Visualisation pour l'Évaluation
Fonctions pour créer des graphiques de comparaison entre les modèles
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import os

# Import de la configuration
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from utils.config import *


# Configuration du style des graphiques
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


# ============================================================================
# COMPARAISON GLOBALE DES MODÈLES
# ============================================================================

def plot_metrics_comparison(results_dict: Dict[str, Dict], 
                           save_path: str = None,
                           title: str = "Comparaison des Performances des Modèles"):
    """
    Crée un histogramme comparant les performances globales des 3 modèles
    
    Ce graphique est le plus important car il montre d'un coup d'œil
    quel modèle performe le mieux sur les métriques principales.
    
    Args:
        results_dict: Dict {model_name: {'global': {'iou': X, 'dice': Y, ...}}}
        save_path: Chemin pour sauvegarder (optionnel)
        title: Titre du graphique
        
    Exemple d'utilisation:
        results = {
            'U-Net': {'global': {'iou': 0.75, 'dice': 0.82, 'pixel_accuracy': 0.88}},
            'YOLO': {'global': {'iou': 0.78, 'dice': 0.85, 'pixel_accuracy': 0.90}},
            'Hybrid': {'global': {'iou': 0.80, 'dice': 0.87, 'pixel_accuracy': 0.91}}
        }
        plot_metrics_comparison(results)
    """
    # Extraire les noms des modèles et les métriques
    model_names = list(results_dict.keys())
    metrics_names = ['IoU', 'Dice', 'Pixel Accuracy']
    
    # Extraire les valeurs
    iou_values = [results_dict[model]['global']['iou'] for model in model_names]
    dice_values = [results_dict[model]['global']['dice'] for model in model_names]
    acc_values = [results_dict[model]['global']['pixel_accuracy'] for model in model_names]
    
    # Configuration du graphique
    x = np.arange(len(metrics_names))
    width = 0.25  # Largeur des barres
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Créer les barres pour chaque modèle
    colors = ['#3498db', '#e74c3c', '#2ecc71']  # Bleu, Rouge, Vert
    
    bars1 = ax.bar(x - width, [iou_values[0], dice_values[0], acc_values[0]], 
                   width, label=model_names[0], color=colors[0], alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x, [iou_values[1], dice_values[1], acc_values[1]], 
                   width, label=model_names[1], color=colors[1], alpha=0.8, edgecolor='black')
    bars3 = ax.bar(x + width, [iou_values[2], dice_values[2], acc_values[2]], 
                   width, label=model_names[2], color=colors[2], alpha=0.8, edgecolor='black')
    
    # Ajouter les valeurs au-dessus des barres
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    add_value_labels(bars1)
    add_value_labels(bars2)
    add_value_labels(bars3)
    
    # Configuration des axes et légendes
    ax.set_xlabel('Métriques', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.legend(fontsize=11)
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Graphique sauvegardé: {save_path}")
    
    plt.show()


# ============================================================================
# MÉTRIQUES PAR CLASSE
# ============================================================================

def plot_per_class_metrics(results_dict: Dict[str, Dict], 
                           metric: str = 'iou',
                           save_path: str = None):
    """
    Crée un graphique montrant les performances par classe pour chaque modèle
    
    Ce graphique est crucial pour identifier quels types de dommages
    sont les plus difficiles à détecter et quel modèle est le meilleur
    pour chaque type de dégradation.
    
    Args:
        results_dict: Dict avec les résultats des modèles
        metric: 'iou' ou 'dice' ou 'f1'
        save_path: Chemin pour sauvegarder
        
    Exemple:
        Si le graphique montre que tous les modèles ont un IoU faible
        pour les fissures crocodiles (classe 2), cela indique que cette
        classe est particulièrement difficile à segmenter.
    """
    model_names = list(results_dict.keys())
    
    # Classes de dommages (on exclut le background pour plus de clarté)
    class_ids = [0, 1, 2, 4]  # Sans le background
    class_labels = [CLASS_NAMES[cid] for cid in class_ids]
    
    # Configuration
    fig, ax = plt.subplots(figsize=(14, 7))
    
    x = np.arange(len(class_ids))
    width = 0.25
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    # Extraire les valeurs par classe pour chaque modèle
    for i, model_name in enumerate(model_names):
        values = []
        for class_id in class_ids:
            # Mapper class_id vers l'index dans per_class (0->1, 1->2, 2->3, 4->4)
            if class_id == 4:
                key = 4
            else:
                key = class_id + 1
            
            value = results_dict[model_name]['per_class'][key][metric]
            values.append(value)
        
        # Créer les barres
        offset = (i - 1) * width
        bars = ax.bar(x + offset, values, width, label=model_name, 
                     color=colors[i], alpha=0.8, edgecolor='black')
        
        # Ajouter les valeurs
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # Configuration
    ax.set_xlabel('Type de Dommage', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{metric.upper()} Score', fontsize=12, fontweight='bold')
    ax.set_title(f'Performances par Classe - {metric.upper()}', 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(class_labels, rotation=15, ha='right')
    ax.legend(fontsize=11)
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Graphique par classe sauvegardé: {save_path}")
    
    plt.show()


# ============================================================================
# MATRICES DE CONFUSION
# ============================================================================

def plot_confusion_matrices(results_dict: Dict[str, Dict], save_path: str = None):
    """
    Affiche les matrices de confusion des 3 modèles côte à côte
    
    La matrice de confusion est essentielle pour comprendre les erreurs
    du modèle. Elle montre quelles classes sont confondues entre elles.
    
    Par exemple, si on voit beaucoup de pixels de la classe "Fissure longitudinale"
    prédits comme "Fissure transversale", cela indique que le modèle a du mal
    à distinguer ces deux types de fissures.
    
    Args:
        results_dict: Dict avec les résultats des modèles
        save_path: Chemin pour sauvegarder
    """
    model_names = list(results_dict.keys())
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Labels pour les classes
    labels = ['BG', 'Long.', 'Trans.', 'Croco.', 'Pothole']
    
    for idx, model_name in enumerate(model_names):
        # Récupérer la matrice de confusion
        conf_matrix = np.array(results_dict[model_name]['confusion_matrix'])
        
        # Normaliser par ligne (pourcentage par classe vraie)
        conf_matrix_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
        conf_matrix_norm = np.nan_to_num(conf_matrix_norm)  # Remplacer NaN par 0
        
        # Créer le heatmap
        sns.heatmap(conf_matrix_norm, 
                   annot=True, 
                   fmt='.2f', 
                   cmap='Blues',
                   xticklabels=labels,
                   yticklabels=labels,
                   ax=axes[idx],
                   cbar_kws={'label': 'Proportion'},
                   vmin=0, vmax=1)
        
        axes[idx].set_title(f'{model_name}', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Classe Prédite', fontsize=10)
        axes[idx].set_ylabel('Classe Vraie', fontsize=10)
    
    plt.suptitle('Matrices de Confusion Normalisées', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Matrices de confusion sauvegardées: {save_path}")
    
    plt.show()


# ============================================================================
# GRILLE DE PRÉDICTIONS
# ============================================================================

def plot_predictions_grid(images: List[np.ndarray],
                         masks_true: List[np.ndarray],
                         predictions_dict: Dict[str, List[np.ndarray]],
                         num_samples: int = 4,
                         save_path: str = None):
    """
    Crée une grille montrant les prédictions des 3 modèles
    
    Cette visualisation est très utile pour comprendre qualitativement
    les différences entre les modèles. On peut voir visuellement où
    chaque modèle performe mieux ou moins bien.
    
    Chaque ligne montre :
    - Colonne 1 : Image originale
    - Colonne 2 : Ground truth (masque vrai)
    - Colonne 3 : Prédiction U-Net
    - Colonne 4 : Prédiction YOLO
    - Colonne 5 : Prédiction Hybrid
    
    Args:
        images: Liste d'images originales
        masks_true: Liste des masques vrais
        predictions_dict: Dict {model_name: [list of predicted masks]}
        num_samples: Nombre d'échantillons à afficher
        save_path: Chemin pour sauvegarder
    """
    num_samples = min(num_samples, len(images))
    model_names = list(predictions_dict.keys())
    
    # 5 colonnes : Image, GT, Modèle1, Modèle2, Modèle3
    fig, axes = plt.subplots(num_samples, 5, figsize=(20, 4*num_samples))
    
    # Si un seul échantillon, axes n'est pas un tableau 2D
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    # Colormap pour les masques
    cmap = plt.cm.get_cmap('tab10')
    
    for row in range(num_samples):
        # Image originale
        axes[row, 0].imshow(images[row])
        axes[row, 0].set_title('Image Originale', fontsize=10, fontweight='bold')
        axes[row, 0].axis('off')
        
        # Ground Truth
        axes[row, 1].imshow(masks_true[row], cmap=cmap, vmin=0, vmax=4)
        axes[row, 1].set_title('Ground Truth', fontsize=10, fontweight='bold')
        axes[row, 1].axis('off')
        
        # Prédictions des modèles
        for col_idx, model_name in enumerate(model_names):
            pred_mask = predictions_dict[model_name][row]
            axes[row, col_idx + 2].imshow(pred_mask, cmap=cmap, vmin=0, vmax=4)
            axes[row, col_idx + 2].set_title(f'{model_name}', fontsize=10, fontweight='bold')
            axes[row, col_idx + 2].axis('off')
    
    plt.suptitle('Comparaison Visuelle des Prédictions', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Grille de prédictions sauvegardée: {save_path}")
    
    plt.show()


# ============================================================================
# COURBES D'ENTRAÎNEMENT
# ============================================================================

def plot_training_curves(history_dict: Dict[str, Dict], save_path: str = None):
    """
    Affiche les courbes de loss et métriques pendant l'entraînement
    
    Ces courbes permettent de comprendre comment les modèles ont appris :
    - Convergence rapide ou lente ?
    - Overfitting (écart entre train et validation) ?
    - Stabilité de l'entraînement ?
    
    Args:
        history_dict: Dict {model_name: {'loss': [...], 'val_loss': [...], ...}}
        save_path: Chemin pour sauvegarder
    """
    model_names = list(history_dict.keys())
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    # Loss
    for idx, model_name in enumerate(model_names):
        history = history_dict[model_name]
        
        if 'loss' in history and 'val_loss' in history:
            axes[0, idx].plot(history['loss'], label='Train Loss', 
                            color=colors[idx], linewidth=2, alpha=0.8)
            axes[0, idx].plot(history['val_loss'], label='Val Loss', 
                            color=colors[idx], linewidth=2, linestyle='--', alpha=0.8)
            axes[0, idx].set_title(f'{model_name} - Loss', fontsize=11, fontweight='bold')
            axes[0, idx].set_xlabel('Epoch')
            axes[0, idx].set_ylabel('Loss')
            axes[0, idx].legend()
            axes[0, idx].grid(alpha=0.3)
    
    # Métriques (IoU ou Accuracy)
    for idx, model_name in enumerate(model_names):
        history = history_dict[model_name]
        
        # Essayer de trouver les métriques disponibles
        metric_key = None
        if 'iou' in history:
            metric_key = 'iou'
        elif 'dice' in history:
            metric_key = 'dice'
        elif 'accuracy' in history:
            metric_key = 'accuracy'
        
        if metric_key and f'val_{metric_key}' in history:
            axes[1, idx].plot(history[metric_key], label=f'Train {metric_key.upper()}', 
                            color=colors[idx], linewidth=2, alpha=0.8)
            axes[1, idx].plot(history[f'val_{metric_key}'], label=f'Val {metric_key.upper()}', 
                            color=colors[idx], linewidth=2, linestyle='--', alpha=0.8)
            axes[1, idx].set_title(f'{model_name} - {metric_key.upper()}', 
                                 fontsize=11, fontweight='bold')
            axes[1, idx].set_xlabel('Epoch')
            axes[1, idx].set_ylabel(metric_key.upper())
            axes[1, idx].legend()
            axes[1, idx].grid(alpha=0.3)
    
    plt.suptitle('Courbes d\'Entraînement', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Courbes d'entraînement sauvegardées: {save_path}")
    
    plt.show()


# ============================================================================
# RAPPORT D'ÉVALUATION COMPLET
# ============================================================================

def create_evaluation_report(results_dict: Dict[str, Dict], 
                            output_dir: str = FIGURES_DIR):
    """
    Crée tous les graphiques d'évaluation en une seule fois
    
    Cette fonction génère automatiquement tous les graphiques nécessaires
    pour l'analyse des performances des modèles. C'est parfait pour créer
    un rapport complet sans avoir à appeler chaque fonction individuellement.
    
    Args:
        results_dict: Dictionnaire avec tous les résultats
        output_dir: Dossier où sauvegarder les graphiques
        
    Graphiques créés:
        1. comparison_global.png - Comparaison globale
        2. comparison_iou_per_class.png - IoU par classe
        3. comparison_dice_per_class.png - Dice par classe
        4. confusion_matrices.png - Matrices de confusion
    """
    print("\n" + "=" * 80)
    print("CRÉATION DU RAPPORT D'ÉVALUATION COMPLET")
    print("=" * 80)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Comparaison globale
    print("\n📊 Création du graphique de comparaison globale...")
    save_path = os.path.join(output_dir, 'comparison_global.png')
    plot_metrics_comparison(results_dict, save_path=save_path)
    
    # 2. IoU par classe
    print("\n📊 Création du graphique IoU par classe...")
    save_path = os.path.join(output_dir, 'comparison_iou_per_class.png')
    plot_per_class_metrics(results_dict, metric='iou', save_path=save_path)
    
    # 3. Dice par classe
    print("\n📊 Création du graphique Dice par classe...")
    save_path = os.path.join(output_dir, 'comparison_dice_per_class.png')
    plot_per_class_metrics(results_dict, metric='dice', save_path=save_path)
    
    # 4. Matrices de confusion
    print("\n📊 Création des matrices de confusion...")
    save_path = os.path.join(output_dir, 'confusion_matrices.png')
    plot_confusion_matrices(results_dict, save_path=save_path)
    
    print("\n" + "=" * 80)
    print("✅ RAPPORT D'ÉVALUATION COMPLET CRÉÉ!")
    print("=" * 80)
    print(f"\n📁 Tous les graphiques sont dans: {output_dir}")
    print("\nGraphiques créés:")
    print("  1. comparison_global.png")
    print("  2. comparison_iou_per_class.png")
    print("  3. comparison_dice_per_class.png")
    print("  4. confusion_matrices.png")


# ============================================================================
# FONCTION DE TEST
# ============================================================================

def test_visualization():
    """
    Fonction de test du module de visualisation
    """
    print("\n" + "=" * 80)
    print("TEST DU MODULE DE VISUALISATION")
    print("=" * 80)
    
    # Créer des résultats fictifs pour tester
    results_dict = {
        'U-Net': {
            'global': {
                'iou': 0.75,
                'dice': 0.82,
                'pixel_accuracy': 0.88
            },
            'per_class': {
                0: {'iou': 0.95, 'dice': 0.97},  # Background
                1: {'iou': 0.72, 'dice': 0.79},  # Longitudinal
                2: {'iou': 0.68, 'dice': 0.75},  # Transverse
                3: {'iou': 0.65, 'dice': 0.72},  # Crocodile
                4: {'iou': 0.70, 'dice': 0.77}   # Pothole
            },
            'confusion_matrix': np.random.randint(0, 1000, (5, 5)).tolist()
        },
        'YOLO': {
            'global': {
                'iou': 0.78,
                'dice': 0.85,
                'pixel_accuracy': 0.90
            },
            'per_class': {
                0: {'iou': 0.96, 'dice': 0.98},
                1: {'iou': 0.75, 'dice': 0.82},
                2: {'iou': 0.71, 'dice': 0.78},
                3: {'iou': 0.68, 'dice': 0.75},
                4: {'iou': 0.73, 'dice': 0.80}
            },
            'confusion_matrix': np.random.randint(0, 1000, (5, 5)).tolist()
        },
        'Hybrid': {
            'global': {
                'iou': 0.80,
                'dice': 0.87,
                'pixel_accuracy': 0.91
            },
            'per_class': {
                0: {'iou': 0.97, 'dice': 0.98},
                1: {'iou': 0.77, 'dice': 0.84},
                2: {'iou': 0.73, 'dice': 0.80},
                3: {'iou': 0.70, 'dice': 0.77},
                4: {'iou': 0.75, 'dice': 0.82}
            },
            'confusion_matrix': np.random.randint(0, 1000, (5, 5)).tolist()
        }
    }
    
    print("\n✅ Résultats de test créés")
    print("\n📊 Test du graphique de comparaison globale...")
    plot_metrics_comparison(results_dict)
    
    print("\n📊 Test du graphique IoU par classe...")
    plot_per_class_metrics(results_dict, metric='iou')
    
    print("\n📊 Test des matrices de confusion...")
    plot_confusion_matrices(results_dict)
    
    print("\n" + "=" * 80)
    print("✅ TOUS LES TESTS DE VISUALISATION SONT PASSÉS!")
    print("=" * 80)


if __name__ == "__main__":
    test_visualization()
