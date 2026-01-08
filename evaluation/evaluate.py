"""
Script Principal d'Évaluation
Évalue et compare les 3 architectures : U-Net, YOLO, Hybrid
"""

import sys
import os

# Ajouter le répertoire parent au path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import json
from typing import Dict
from datetime import datetime

# Imports du projet
from preprocessing.data_loader import RDD2022DataLoader
from preprocessing.preprocessing import ImagePreprocessor
from evaluation.metrics import evaluate_model_on_dataset
from evaluation.visualization import create_evaluation_report, plot_predictions_grid
from utils.config import *
from utils.helpers import create_experiment_folder, save_results_json, print_summary_table

# Imports pour charger les modèles
import tensorflow as tf
from tensorflow import keras


# ============================================================================
# CHARGEMENT DES MODÈLES
# ============================================================================

def load_trained_models(model_paths: Dict[str, str]) -> Dict:
    """
    Charge les 3 modèles entraînés depuis le disque
    
    Cette fonction charge les modèles que tu as déjà entraînés.
    Elle gère les différents formats (Keras .h5 pour U-Net et Hybrid,
    et YOLO pour le modèle YOLO).
    
    Args:
        model_paths: Dict {model_name: path_to_model}
        Exemple:
        {
            'U-Net': 'results/models/unet_best.h5',
            'YOLO': 'results/models/yolo_best.pt',
            'Hybrid': 'results/models/hybrid_best.h5'
        }
        
    Returns:
        Dict {model_name: loaded_model}
    """
    print("\n" + "=" * 100)
    print("CHARGEMENT DES MODÈLES ENTRAÎNÉS")
    print("=" * 100)
    
    models = {}
    
    for model_name, model_path in model_paths.items():
        print(f"\n📦 Chargement de {model_name}...")
        
        if not os.path.exists(model_path):
            print(f"   ⚠️  Fichier non trouvé: {model_path}")
            print(f"   ⚠️  Ce modèle sera ignoré dans l'évaluation")
            continue
        
        try:
            if model_name == 'YOLO':
                # Charger le modèle YOLO (Ultralytics)
                try:
                    from ultralytics import YOLO
                    model = YOLO(model_path)
                    print(f"   ✅ YOLO chargé depuis: {model_path}")
                except ImportError:
                    print(f"   ❌ Ultralytics non disponible. Installez avec: pip install ultralytics")
                    continue
            else:
                # Charger les modèles Keras (U-Net et Hybrid)
                from models.model_utils import DiceCoefficient, IoUMetric
                
                # Custom objects pour les métriques personnalisées
                custom_objects = {
                    'DiceCoefficient': DiceCoefficient,
                    'IoUMetric': IoUMetric,
                    'dice_coefficient': DiceCoefficient,
                    'iou_metric': IoUMetric
                }
                
                model = keras.models.load_model(model_path, custom_objects=custom_objects)
                print(f"   ✅ {model_name} chargé depuis: {model_path}")
            
            models[model_name] = model
            
        except Exception as e:
            print(f"   ❌ Erreur lors du chargement de {model_name}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n✅ {len(models)} modèles chargés avec succès")
    return models


# ============================================================================
# PRÉDICTION AVEC LES MODÈLES
# ============================================================================

def predict_with_model(model, 
                      model_name: str,
                      images: np.ndarray,
                      preprocessor: ImagePreprocessor) -> np.ndarray:
    """
    Effectue des prédictions avec un modèle donné
    
    Cette fonction gère les différences entre les modèles :
    - U-Net et Hybrid : prédiction directe avec Keras
    - YOLO : prédiction puis conversion du format YOLO vers notre format
    
    Args:
        model: Le modèle (Keras ou YOLO)
        model_name: Nom du modèle ('U-Net', 'YOLO', 'Hybrid')
        images: Batch d'images prétraitées (N, H, W, C)
        preprocessor: Préprocesseur pour la normalisation
        
    Returns:
        Masques prédits (N, H, W) avec class IDs
    """
    if model_name == 'YOLO':
        # Prédiction YOLO
        predictions = []
        
        for img in images:
            # Dénormaliser l'image pour YOLO
            img_uint8 = preprocessor.denormalize_image(img)
            
            # Prédire
            results = model.predict(source=img_uint8, conf=0.25, save=False, verbose=False)
            
            # Convertir les masques YOLO en format standard
            from models.yolo_pretrained import YOLOSegmentation
            yolo_wrapper = YOLOSegmentation()
            mask = yolo_wrapper.convert_masks_to_segmentation(results, target_size=IMG_SIZE)
            
            predictions.append(mask)
        
        return np.array(predictions)
    
    else:
        # Prédiction Keras (U-Net et Hybrid)
        predictions = model.predict(images, verbose=0)
        
        # Convertir les probabilités en classes
        masks = np.argmax(predictions, axis=-1)
        
        return masks


# ============================================================================
# ÉVALUATION COMPLÈTE DES MODÈLES
# ============================================================================

def evaluate_all_models(data_path: str,
                       model_paths: Dict[str, str],
                       num_samples: int = None,
                       batch_size: int = 8,
                       save_predictions: bool = True) -> Dict:
    """
    Fonction principale d'évaluation
    
    Cette fonction orchestre toute l'évaluation :
    1. Charge les modèles entraînés
    2. Charge le dataset de test
    3. Évalue chaque modèle
    4. Calcule toutes les métriques
    5. Génère tous les graphiques
    6. Sauvegarde les résultats
    
    C'est la fonction que tu vas appeler pour lancer l'évaluation complète.
    
    Args:
        data_path: Chemin vers RDD_SPLIT
        model_paths: Dict {model_name: path_to_model}
        num_samples: Nombre d'échantillons à évaluer (None = tous)
        batch_size: Taille des batches
        save_predictions: Si True, sauvegarde des exemples de prédictions
        
    Returns:
        Dict avec tous les résultats
    """
    print("\n" + "🎯 " * 40)
    print("ÉVALUATION COMPLÈTE DES MODÈLES DE SEGMENTATION")
    print("🎯 " * 40)
    
    # ========================================================================
    # 1. PRÉPARATION
    # ========================================================================
    print("\n📦 PHASE 1: Préparation")
    print("-" * 100)
    
    # Créer un dossier pour cette évaluation
    eval_folder = create_experiment_folder("evaluation")
    print(f"✅ Dossier d'évaluation créé: {eval_folder}")
    
    # Charger les modèles
    models = load_trained_models(model_paths)
    
    if len(models) == 0:
        print("\n❌ Aucun modèle n'a pu être chargé. Vérifiez les chemins.")
        return {}
    
    # Charger le dataset de test
    print(f"\n📂 Chargement du dataset de test...")
    test_loader = RDD2022DataLoader(data_path, split='test')
    print(f"✅ Dataset de test: {len(test_loader)} images")
    
    # Créer le préprocesseur
    preprocessor = ImagePreprocessor(target_size=IMG_SIZE, normalize=True)
    
    # ========================================================================
    # 2. ÉVALUATION DE CHAQUE MODÈLE
    # ========================================================================
    print("\n📊 PHASE 2: Évaluation des modèles")
    print("-" * 100)
    
    results_dict = {}
    predictions_dict = {}
    
    for model_name, model in models.items():
        print(f"\n{'=' * 100}")
        print(f"ÉVALUATION DE {model_name.upper()}")
        print(f"{'=' * 100}")
        
        # Évaluer le modèle
        results = evaluate_model_on_dataset(
            model=model,
            data_loader=test_loader,
            preprocessor=preprocessor,
            num_samples=num_samples,
            batch_size=batch_size,
            model_name=model_name
        )
        
        results_dict[model_name] = results
        
        # Sauvegarder quelques prédictions pour visualisation
        if save_predictions:
            print(f"\n💾 Génération de prédictions pour visualisation...")
            sample_predictions = []
            
            for idx in range(min(10, len(test_loader))):
                image, mask, _ = test_loader[idx]
                proc_image = preprocessor.preprocess_image(image)
                
                # Prédire
                pred_mask = predict_with_model(
                    model, 
                    model_name, 
                    np.array([proc_image]), 
                    preprocessor
                )[0]
                
                sample_predictions.append(pred_mask)
            
            predictions_dict[model_name] = sample_predictions
    
    # ========================================================================
    # 3. COMPARAISON ET VISUALISATION
    # ========================================================================
    print("\n📊 PHASE 3: Génération des graphiques de comparaison")
    print("-" * 100)
    
    figures_dir = os.path.join(eval_folder, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    # Créer tous les graphiques
    create_evaluation_report(results_dict, output_dir=figures_dir)
    
    # Créer la grille de prédictions si on a sauvegardé les prédictions
    if save_predictions and predictions_dict:
        print("\n📊 Création de la grille de prédictions...")
        
        # Charger quelques images de test
        sample_images = []
        sample_masks = []
        
        for idx in range(min(4, len(test_loader))):
            image, mask, _ = test_loader[idx]
            sample_images.append(image)
            sample_masks.append(preprocessor.preprocess_mask(mask))
        
        # Créer la grille
        save_path = os.path.join(figures_dir, 'predictions_grid.png')
        plot_predictions_grid(
            images=sample_images,
            masks_true=sample_masks,
            predictions_dict=predictions_dict,
            num_samples=min(4, len(sample_images)),
            save_path=save_path
        )
    
    # ========================================================================
    # 4. SAUVEGARDE DES RÉSULTATS
    # ========================================================================
    print("\n💾 PHASE 4: Sauvegarde des résultats")
    print("-" * 100)
    
    # Sauvegarder en JSON
    results_file = os.path.join(eval_folder, 'evaluation_results.json')
    save_results_json(results_dict, results_file)
    
    # Créer un résumé texte
    summary_file = os.path.join(eval_folder, 'evaluation_summary.txt')
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write("RÉSUMÉ DE L'ÉVALUATION DES MODÈLES\n")
        f.write("=" * 100 + "\n\n")
        
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Dataset: {data_path}\n")
        f.write(f"Nombre d'images évaluées: {num_samples or len(test_loader)}\n\n")
        
        # Tableau comparatif
        f.write("PERFORMANCES GLOBALES\n")
        f.write("-" * 100 + "\n\n")
        
        # En-tête
        f.write(f"{'Modèle':<15} {'IoU':<10} {'Dice':<10} {'Pixel Accuracy':<15}\n")
        f.write("-" * 100 + "\n")
        
        # Lignes
        for model_name, results in results_dict.items():
            iou = results['global']['iou']
            dice = results['global']['dice']
            acc = results['global']['pixel_accuracy']
            f.write(f"{model_name:<15} {iou:<10.4f} {dice:<10.4f} {acc:<15.4f}\n")
        
        f.write("\n" + "=" * 100 + "\n")
        
        # Métriques par classe
        f.write("\nPERFORMANCES PAR CLASSE\n")
        f.write("-" * 100 + "\n\n")
        
        for model_name, results in results_dict.items():
            f.write(f"\n{model_name}:\n")
            f.write("-" * 50 + "\n")
            
            for class_id, metrics in results['per_class'].items():
                class_name = CLASS_NAMES.get(class_id - 1 if class_id > 0 else 0, f"Classe {class_id}")
                if class_id == 0:
                    class_name = "Background"
                
                f.write(f"  {class_name:<20} IoU: {metrics['iou']:.4f}  Dice: {metrics['dice']:.4f}\n")
    
    print(f"✅ Résultats sauvegardés: {results_file}")
    print(f"✅ Résumé sauvegardé: {summary_file}")
    
    # ========================================================================
    # 5. AFFICHAGE DU TABLEAU RÉCAPITULATIF
    # ========================================================================
    print("\n" + "=" * 100)
    print("TABLEAU RÉCAPITULATIF DES PERFORMANCES")
    print("=" * 100)
    
    # Préparer les données pour le tableau
    comparison_data = {}
    for model_name, results in results_dict.items():
        comparison_data[model_name] = {
            'IoU': results['global']['iou'],
            'Dice': results['global']['dice'],
            'Pixel Accuracy': results['global']['pixel_accuracy']
        }
    
    print_summary_table(comparison_data)
    
    # Identifier le meilleur modèle
    best_model_iou = max(results_dict.items(), key=lambda x: x[1]['global']['iou'])
    best_model_dice = max(results_dict.items(), key=lambda x: x[1]['global']['dice'])
    
    print("\n🏆 MEILLEURS MODÈLES:")
    print(f"  - Meilleur IoU: {best_model_iou[0]} ({best_model_iou[1]['global']['iou']:.4f})")
    print(f"  - Meilleur Dice: {best_model_dice[0]} ({best_model_dice[1]['global']['dice']:.4f})")
    
    print("\n" + "=" * 100)
    print("✅ ÉVALUATION COMPLÈTE TERMINÉE !")
    print("=" * 100)
    
    print(f"\n📁 Tous les résultats sont dans: {eval_folder}")
    print("\n📊 Graphiques créés:")
    print("  - comparison_global.png")
    print("  - comparison_iou_per_class.png")
    print("  - comparison_dice_per_class.png")
    print("  - confusion_matrices.png")
    print("  - predictions_grid.png")
    
    return results_dict


# ============================================================================
# FONCTIONS UTILITAIRES
# ============================================================================

def compare_models(results_dict: Dict) -> str:
    """
    Compare les modèles et génère un rapport textuel
    
    Cette fonction analyse les résultats et génère des insights
    sur les forces et faiblesses de chaque modèle.
    
    Args:
        results_dict: Dict avec les résultats de tous les modèles
        
    Returns:
        String avec le rapport d'analyse
    """
    report = []
    report.append("\n" + "=" * 100)
    report.append("ANALYSE COMPARATIVE DES MODÈLES")
    report.append("=" * 100 + "\n")
    
    # Comparaison globale
    report.append("1. PERFORMANCES GLOBALES")
    report.append("-" * 100)
    
    for model_name, results in results_dict.items():
        iou = results['global']['iou']
        dice = results['global']['dice']
        acc = results['global']['pixel_accuracy']
        
        report.append(f"\n{model_name}:")
        report.append(f"  - IoU: {iou:.4f}")
        report.append(f"  - Dice: {dice:.4f}")
        report.append(f"  - Accuracy: {acc:.4f}")
    
    # Analyse par classe
    report.append("\n\n2. ANALYSE PAR TYPE DE DOMMAGE")
    report.append("-" * 100)
    
    # Trouver les classes les plus difficiles
    class_difficulties = {cid: [] for cid in range(NUM_CLASSES + 1)}
    
    for model_name, results in results_dict.items():
        for class_id, metrics in results['per_class'].items():
            class_difficulties[class_id].append((model_name, metrics['iou']))
    
    for class_id, performances in class_difficulties.items():
        if class_id == 0:
            class_name = "Background"
        else:
            class_name = CLASS_NAMES.get(class_id - 1, f"Classe {class_id}")
        
        avg_iou = np.mean([p[1] for p in performances])
        best_model = max(performances, key=lambda x: x[1])
        
        report.append(f"\n{class_name}:")
        report.append(f"  - IoU moyen: {avg_iou:.4f}")
        report.append(f"  - Meilleur modèle: {best_model[0]} (IoU: {best_model[1]:.4f})")
        
        if avg_iou < 0.5:
            report.append(f"  ⚠️  Classe difficile (IoU < 0.5)")
    
    report.append("\n" + "=" * 100)
    
    return "\n".join(report)


def save_evaluation_results(results_dict: Dict, output_path: str):
    """
    Sauvegarde les résultats d'évaluation en JSON
    
    Args:
        results_dict: Résultats de l'évaluation
        output_path: Chemin de sauvegarde
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, indent=4, ensure_ascii=False)
    
    print(f"✅ Résultats sauvegardés: {output_path}")


# ============================================================================
# SCRIPT PRINCIPAL
# ============================================================================

if __name__ == "__main__":
    """
    Script principal d'évaluation
    
    Pour utiliser ce script :
    1. Assure-toi que tes modèles sont entraînés et sauvegardés
    2. Modifie les chemins vers tes modèles ci-dessous
    3. Lance le script : python evaluation/evaluate.py
    """
    
    # Configuration
    DATA_PATH = "C:/Users/DELL/Desktop/dataset/RDD_SPLIT"
    
    # Chemins vers les modèles entraînés
    # ⚠️ MODIFIE CES CHEMINS SELON TES MODÈLES ENTRAÎNÉS
    MODEL_PATHS = {
        'U-Net': 'C:/Users/DELL/Desktop/results/models/unet_100img_20260103_231900.keras',
        'YOLO': 'C:/Users/DELL/Desktop/dataset/yolo_temp_results/yolo_100img2/weights/last.pt',  # Chemin YOLO typique
        'Hybrid': 'C:/Users/DELL/Desktop/results/models/hybrid_100img_20260104_111932.keras'
    }
    
    # Paramètres d'évaluation
    NUM_SAMPLES = 100  # None = tout le dataset de test
    BATCH_SIZE = 8
    
    # Lancer l'évaluation complète
    results = evaluate_all_models(
        data_path=DATA_PATH,
        model_paths=MODEL_PATHS,
        num_samples=NUM_SAMPLES,
        batch_size=BATCH_SIZE,
        save_predictions=True
    )
    
    # Afficher l'analyse comparative
    if results:
        analysis = compare_models(results)
        print(analysis)
