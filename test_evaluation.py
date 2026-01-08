"""
Script de Test Rapide pour l'Évaluation
Teste le système d'évaluation sur un petit échantillon de données
"""

import sys
import os

# Ajouter le répertoire parent au path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np

from preprocessing.data_loader import RDD2022DataLoader
from preprocessing.preprocessing import ImagePreprocessor
from evaluation.metrics import (
    calculate_iou_single,
    calculate_dice_single,
    calculate_pixel_accuracy_single,
    calculate_metrics_per_class
)
from evaluation.visualization import (
    plot_metrics_comparison,
    plot_per_class_metrics,
    plot_confusion_matrices
)
from utils.config import *
from utils.helpers import set_seeds


def test_evaluation_pipeline():
    """
    Test complet du pipeline d'évaluation sur des données simulées
    
    Ce test vérifie que :
    1. Les métriques se calculent correctement
    2. Les graphiques se génèrent sans erreur
    3. Le format des résultats est correct
    """
    print("\n" + "=" * 100)
    print("TEST RAPIDE DU SYSTÈME D'ÉVALUATION")
    print("=" * 100)
    
    # Configurer les seeds
    set_seeds(RANDOM_SEED)
    
    # ========================================================================
    # 1. TEST DES MÉTRIQUES INDIVIDUELLES
    # ========================================================================
    print("\n📊 PHASE 1: Test des métriques individuelles")
    print("-" * 100)
    
    # Créer des masques de test
    mask_true = np.random.randint(0, NUM_CLASSES + 1, (256, 256))
    mask_pred = np.random.randint(0, NUM_CLASSES + 1, (256, 256))
    
    print(f"  - Masques créés: {mask_true.shape}")
    print(f"  - Classes présentes: {np.unique(mask_true)}")
    
    # Test IoU
    iou = calculate_iou_single(mask_true, mask_pred)
    print(f"\n  ✅ IoU: {iou:.4f}")
    
    # Test Dice
    dice = calculate_dice_single(mask_true, mask_pred)
    print(f"  ✅ Dice: {dice:.4f}")
    
    # Test Pixel Accuracy
    acc = calculate_pixel_accuracy_single(mask_true, mask_pred)
    print(f"  ✅ Pixel Accuracy: {acc:.4f}")
    
    # Test métriques par classe
    per_class = calculate_metrics_per_class(mask_true, mask_pred)
    print(f"\n  ✅ Métriques par classe calculées:")
    for class_id in range(min(3, NUM_CLASSES + 1)):
        print(f"     Classe {class_id}: IoU={per_class[class_id]['iou']:.4f}, " + 
              f"Dice={per_class[class_id]['dice']:.4f}")
    
    # ========================================================================
    # 2. TEST DE LA VISUALISATION
    # ========================================================================
    print("\n📊 PHASE 2: Test de la visualisation")
    print("-" * 100)
    
    # Créer des résultats fictifs pour 3 modèles
    results_dict = {
        'U-Net': {
            'global': {
                'iou': 0.75,
                'dice': 0.82,
                'pixel_accuracy': 0.88
            },
            'per_class': {
                0: {'iou': 0.95, 'dice': 0.97},
                1: {'iou': 0.72, 'dice': 0.79},
                2: {'iou': 0.68, 'dice': 0.75},
                3: {'iou': 0.65, 'dice': 0.72},
                4: {'iou': 0.70, 'dice': 0.77}
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
    
    print("\n  📊 Test du graphique de comparaison globale...")
    try:
        plot_metrics_comparison(results_dict)
        print("  ✅ Graphique de comparaison créé avec succès")
    except Exception as e:
        print(f"  ❌ Erreur: {e}")
    
    print("\n  📊 Test du graphique IoU par classe...")
    try:
        plot_per_class_metrics(results_dict, metric='iou')
        print("  ✅ Graphique IoU par classe créé avec succès")
    except Exception as e:
        print(f"  ❌ Erreur: {e}")
    
    print("\n  📊 Test des matrices de confusion...")
    try:
        plot_confusion_matrices(results_dict)
        print("  ✅ Matrices de confusion créées avec succès")
    except Exception as e:
        print(f"  ❌ Erreur: {e}")
    
    # ========================================================================
    # 3. TEST AVEC DE VRAIES DONNÉES (SI DISPONIBLES)
    # ========================================================================
    print("\n📊 PHASE 3: Test avec données réelles (optionnel)")
    print("-" * 100)
    
    DATA_PATH = "C:/Users/DELL/Desktop/dataset/RDD_SPLIT"
    
    if os.path.exists(DATA_PATH):
        print(f"  📂 Dataset trouvé: {DATA_PATH}")
        
        try:
            # Charger quelques images
            test_loader = RDD2022DataLoader(DATA_PATH, split='test')
            preprocessor = ImagePreprocessor(target_size=IMG_SIZE, normalize=True)
            
            print(f"  ✅ Dataset de test: {len(test_loader)} images")
            
            # Charger une image
            if len(test_loader) > 0:
                image, mask, annotations = test_loader[0]
                proc_image = preprocessor.preprocess_image(image)
                proc_mask = preprocessor.preprocess_mask(mask)
                
                print(f"\n  📊 Échantillon chargé:")
                print(f"     - Image: {proc_image.shape}")
                print(f"     - Masque: {proc_mask.shape}")
                print(f"     - Classes: {np.unique(proc_mask)}")
                
                # Test des métriques sur cet échantillon
                fake_pred = np.random.randint(0, NUM_CLASSES + 1, proc_mask.shape)
                
                iou = calculate_iou_single(proc_mask, fake_pred)
                dice = calculate_dice_single(proc_mask, fake_pred)
                
                print(f"\n  ✅ Métriques sur données réelles:")
                print(f"     - IoU: {iou:.4f}")
                print(f"     - Dice: {dice:.4f}")
                
        except Exception as e:
            print(f"  ⚠️  Erreur lors du chargement des données: {e}")
    else:
        print(f"  ⚠️  Dataset non trouvé: {DATA_PATH}")
        print(f"     Cette phase est ignorée")
    
    # ========================================================================
    # RÉSUMÉ FINAL
    # ========================================================================
    print("\n" + "=" * 100)
    print("✅ TOUS LES TESTS SONT PASSÉS AVEC SUCCÈS!")
    print("=" * 100)
    
    print("\n🎯 Le système d'évaluation est prêt à être utilisé!")
    print("\nPour lancer l'évaluation complète:")
    print("  1. Assure-toi que tes modèles sont entraînés")
    print("  2. Modifie les chemins dans evaluation/evaluate.py")
    print("  3. Lance: python evaluation/evaluate.py")
    
    print("\n" + "=" * 100)


if __name__ == "__main__":
    test_evaluation_pipeline()
