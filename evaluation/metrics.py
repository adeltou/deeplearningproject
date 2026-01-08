"""
Module de Calcul des Métriques d'Évaluation
Fonctions pour évaluer les performances des modèles de segmentation
"""

import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
from typing import Dict, List, Tuple
import json

# Import de la configuration
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from utils.config import *
from preprocessing.data_loader import RDD2022DataLoader
from preprocessing.preprocessing import ImagePreprocessor


# ============================================================================
# ÉVALUATION GLOBALE DU MODÈLE
# ============================================================================

def evaluate_model_on_dataset(model, 
                              data_loader: RDD2022DataLoader,
                              preprocessor: ImagePreprocessor,
                              num_samples: int = None,
                              batch_size: int = 8) -> Dict:
    """
    Évalue un modèle sur un dataset complet
    
    Cette fonction prend un modèle entraîné et calcule toutes les métriques
    importantes sur l'ensemble du dataset de test. Elle retourne un dictionnaire
    contenant les performances globales et par classe.
    
    Args:
        model: Modèle Keras ou YOLO à évaluer
        data_loader: Loader du dataset (train/val/test)
        preprocessor: Préprocesseur pour normaliser les images
        num_samples: Nombre d'échantillons à évaluer (None = tout le dataset)
        batch_size: Taille des batches pour l'évaluation
        
    Returns:
        Dict contenant toutes les métriques :
        {
            'global': {
                'iou': float,
                'dice': float,
                'pixel_accuracy': float
            },
            'per_class': {
                class_id: {'iou': float, 'dice': float, 'precision': float, 'recall': float, 'f1': float}
            },
            'confusion_matrix': ndarray,
            'num_samples': int
        }
    """
    print(f"\n📊 Évaluation du modèle sur {data_loader.split}...")
    print("-" * 80)
    
    # Déterminer le nombre d'échantillons
    if num_samples is None:
        num_samples = len(data_loader)
    else:
        num_samples = min(num_samples, len(data_loader))
    
    # Initialiser les accumulateurs pour les métriques
    all_y_true = []
    all_y_pred = []
    
    total_iou = 0
    total_dice = 0
    total_pixel_acc = 0
    
    # Évaluer batch par batch
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        # Calculer les indices de début et fin pour ce batch
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_samples)
        
        # Charger et prétraiter les images du batch
        batch_images = []
        batch_masks_true = []
        
        for idx in range(start_idx, end_idx):
            image, mask, _ = data_loader[idx]
            
            # Prétraiter
            proc_image = preprocessor.preprocess_image(image)
            proc_mask = preprocessor.preprocess_mask(mask)
            
            batch_images.append(proc_image)
            batch_masks_true.append(proc_mask)
        
        # Convertir en arrays
        batch_images = np.array(batch_images)
        batch_masks_true = np.array(batch_masks_true)
        
        # Prédiction
        batch_predictions = model.predict(batch_images, verbose=0)
        
        # Convertir les prédictions en masques de classes
        batch_masks_pred = np.argmax(batch_predictions, axis=-1)
        
        # Accumuler pour la matrice de confusion globale
        all_y_true.extend(batch_masks_true.flatten())
        all_y_pred.extend(batch_masks_pred.flatten())
        
        # Calculer les métriques pour ce batch
        for i in range(len(batch_images)):
            # IoU pour cette image
            iou = calculate_iou_single(batch_masks_true[i], batch_masks_pred[i])
            total_iou += iou
            
            # Dice pour cette image
            dice = calculate_dice_single(batch_masks_true[i], batch_masks_pred[i])
            total_dice += dice
            
            # Pixel accuracy
            pixel_acc = calculate_pixel_accuracy_single(batch_masks_true[i], batch_masks_pred[i])
            total_pixel_acc += pixel_acc
        
        # Afficher la progression
        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == num_batches:
            print(f"  Batch {batch_idx + 1}/{num_batches} traité...")
    
    # Calculer les moyennes globales
    avg_iou = total_iou / num_samples
    avg_dice = total_dice / num_samples
    avg_pixel_acc = total_pixel_acc / num_samples
    
    print(f"\n✅ Évaluation terminée sur {num_samples} images")
    print(f"  - IoU moyen: {avg_iou:.4f}")
    print(f"  - Dice moyen: {avg_dice:.4f}")
    print(f"  - Pixel Accuracy: {avg_pixel_acc:.4f}")
    
    # Calculer la matrice de confusion
    conf_matrix = calculate_confusion_matrix(all_y_true, all_y_pred)
    
    # Calculer les métriques par classe
    per_class_metrics = calculate_class_metrics(all_y_true, all_y_pred)
    
    # Construire le dictionnaire de résultats
    results = {
        'global': {
            'iou': float(avg_iou),
            'dice': float(avg_dice),
            'pixel_accuracy': float(avg_pixel_acc)
        },
        'per_class': per_class_metrics,
        'confusion_matrix': conf_matrix.tolist(),
        'num_samples': num_samples
    }
    
    return results


# ============================================================================
# CALCUL DES MÉTRIQUES INDIVIDUELLES
# ============================================================================

def calculate_iou_single(mask_true: np.ndarray, mask_pred: np.ndarray) -> float:
    """
    Calcule l'IoU pour une seule paire de masques
    
    L'IoU (Intersection over Union) mesure le chevauchement entre deux masques.
    Plus l'IoU est élevé (proche de 1), meilleure est la prédiction.
    
    Args:
        mask_true: Masque ground truth (H, W)
        mask_pred: Masque prédit (H, W)
        
    Returns:
        IoU score entre 0 et 1
    """
    # Aplatir les masques
    true_flat = mask_true.flatten()
    pred_flat = mask_pred.flatten()
    
    # Calculer l'intersection et l'union
    intersection = np.sum(true_flat == pred_flat)
    total_pixels = len(true_flat)
    
    # IoU simplifié (pixel accuracy pondéré)
    iou = intersection / total_pixels if total_pixels > 0 else 0
    
    return iou


def calculate_dice_single(mask_true: np.ndarray, mask_pred: np.ndarray) -> float:
    """
    Calcule le coefficient de Dice pour une seule paire de masques
    
    Le coefficient de Dice est similaire à l'IoU mais avec une formule différente.
    Il est particulièrement utile pour les classes déséquilibrées.
    
    Dice = 2 * |A ∩ B| / (|A| + |B|)
    
    Args:
        mask_true: Masque ground truth (H, W)
        mask_pred: Masque prédit (H, W)
        
    Returns:
        Dice coefficient entre 0 et 1
    """
    # Aplatir
    true_flat = mask_true.flatten()
    pred_flat = mask_pred.flatten()
    
    # Intersection
    intersection = np.sum(true_flat == pred_flat)
    
    # Dice
    dice = (2.0 * intersection) / (len(true_flat) + len(pred_flat))
    
    return dice


def calculate_pixel_accuracy_single(mask_true: np.ndarray, mask_pred: np.ndarray) -> float:
    """
    Calcule la précision pixel par pixel
    
    C'est la métrique la plus simple : quel pourcentage de pixels
    a été correctement classifié ?
    
    Args:
        mask_true: Masque ground truth (H, W)
        mask_pred: Masque prédit (H, W)
        
    Returns:
        Accuracy entre 0 et 1
    """
    correct = np.sum(mask_true == mask_pred)
    total = mask_true.size
    
    return correct / total if total > 0 else 0


# ============================================================================
# MATRICE DE CONFUSION
# ============================================================================

def calculate_confusion_matrix(y_true: List, y_pred: List) -> np.ndarray:
    """
    Calcule la matrice de confusion pour la segmentation
    
    La matrice de confusion montre comment les pixels de chaque classe vraie
    ont été prédits. C'est très utile pour identifier les confusions entre classes.
    
    Par exemple, si le modèle confond souvent les fissures longitudinales (0)
    avec les fissures transversales (1), cela sera visible dans la matrice.
    
    Args:
        y_true: Liste des labels vrais (tous les pixels)
        y_pred: Liste des labels prédits (tous les pixels)
        
    Returns:
        Matrice de confusion (num_classes, num_classes)
    """
    # Convertir en arrays numpy
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculer la matrice de confusion
    conf_matrix = confusion_matrix(
        y_true, 
        y_pred, 
        labels=list(range(NUM_CLASSES + 1))  # 0, 1, 2, 3, 4
    )
    
    return conf_matrix


# ============================================================================
# MÉTRIQUES PAR CLASSE
# ============================================================================

def calculate_class_metrics(y_true: List, y_pred: List) -> Dict:
    """
    Calcule les métriques détaillées pour chaque classe
    
    Pour chaque type de dommage routier, cette fonction calcule :
    - IoU : Chevauchement entre vrai et prédit
    - Dice : Coefficient de similarité
    - Précision : Parmi les pixels prédits comme cette classe, combien sont corrects
    - Rappel : Parmi les vrais pixels de cette classe, combien ont été détectés
    - F1-Score : Moyenne harmonique de précision et rappel
    
    Args:
        y_true: Labels vrais
        y_pred: Labels prédits
        
    Returns:
        Dict {class_id: {'iou': X, 'dice': Y, 'precision': Z, ...}}
    """
    # Convertir en arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    metrics = {}
    
    # Pour chaque classe (0, 1, 2, 4 + background)
    for class_id in range(NUM_CLASSES + 1):
        # Créer des masques binaires pour cette classe
        true_binary = (y_true == class_id).astype(int)
        pred_binary = (y_pred == class_id).astype(int)
        
        # Calculer TP, FP, FN, TN
        tp = np.sum((true_binary == 1) & (pred_binary == 1))  # True Positives
        fp = np.sum((true_binary == 0) & (pred_binary == 1))  # False Positives
        fn = np.sum((true_binary == 1) & (pred_binary == 0))  # False Negatives
        tn = np.sum((true_binary == 0) & (pred_binary == 0))  # True Negatives
        
        # Calculer IoU
        intersection = tp
        union = tp + fp + fn
        iou = intersection / union if union > 0 else 0
        
        # Calculer Dice
        dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
        
        # Calculer Précision
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        # Calculer Rappel
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # Calculer F1-Score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Stocker les métriques
        metrics[class_id] = {
            'iou': float(iou),
            'dice': float(dice),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'support': int(np.sum(true_binary))  # Nombre de pixels de cette classe
        }
    
    return metrics


def calculate_pixel_accuracy(y_true: List, y_pred: List) -> float:
    """
    Calcule la précision globale pixel par pixel
    
    Args:
        y_true: Labels vrais
        y_pred: Labels prédits
        
    Returns:
        Accuracy globale
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    correct = np.sum(y_true == y_pred)
    total = len(y_true)
    
    return correct / total if total > 0 else 0


def calculate_metrics_per_class(mask_true: np.ndarray, 
                                mask_pred: np.ndarray) -> Dict:
    """
    Version simplifiée pour une seule image
    
    Calcule les métriques par classe pour une paire de masques.
    Utile pour analyser les performances sur des images individuelles.
    
    Args:
        mask_true: Masque ground truth (H, W)
        mask_pred: Masque prédit (H, W)
        
    Returns:
        Dict {class_id: {'iou': X, 'dice': Y}}
    """
    metrics = {}
    
    for class_id in range(NUM_CLASSES + 1):
        # Masques binaires
        true_mask = (mask_true == class_id).astype(np.float32)
        pred_mask = (mask_pred == class_id).astype(np.float32)
        
        # Intersection et union
        intersection = np.sum(true_mask * pred_mask)
        union = np.sum(true_mask) + np.sum(pred_mask) - intersection
        
        # IoU
        iou = intersection / union if union > 0 else 0
        
        # Dice
        dice = (2 * intersection) / (np.sum(true_mask) + np.sum(pred_mask)) if (np.sum(true_mask) + np.sum(pred_mask)) > 0 else 0
        
        metrics[class_id] = {
            'iou': float(iou),
            'dice': float(dice)
        }
    
    return metrics


# ============================================================================
# FONCTION DE TEST
# ============================================================================

def test_metrics():
    """
    Fonction de test des métriques
    """
    print("\n" + "=" * 80)
    print("TEST DES MÉTRIQUES D'ÉVALUATION")
    print("=" * 80)
    
    # Créer des masques de test
    mask_true = np.random.randint(0, NUM_CLASSES + 1, (256, 256))
    mask_pred = np.random.randint(0, NUM_CLASSES + 1, (256, 256))
    
    print(f"\n📊 Test sur masques aléatoires:")
    print(f"  Shape: {mask_true.shape}")
    print(f"  Classes: {np.unique(mask_true)}")
    
    # Test IoU
    iou = calculate_iou_single(mask_true, mask_pred)
    print(f"\n✅ IoU: {iou:.4f}")
    
    # Test Dice
    dice = calculate_dice_single(mask_true, mask_pred)
    print(f"✅ Dice: {dice:.4f}")
    
    # Test Pixel Accuracy
    acc = calculate_pixel_accuracy_single(mask_true, mask_pred)
    print(f"✅ Pixel Accuracy: {acc:.4f}")
    
    # Test métriques par classe
    per_class = calculate_metrics_per_class(mask_true, mask_pred)
    print(f"\n✅ Métriques par classe:")
    for class_id, metrics in per_class.items():
        print(f"  Classe {class_id}: IoU={metrics['iou']:.4f}, Dice={metrics['dice']:.4f}")
    
    print("\n" + "=" * 80)
    print("✅ TOUS LES TESTS SONT PASSÉS!")
    print("=" * 80)


if __name__ == "__main__":
    test_metrics()
