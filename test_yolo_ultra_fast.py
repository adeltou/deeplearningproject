"""
Test ULTRA-RAPIDE pour YOLO Segmentation
Utilise seulement 200 images pour tester que tout fonctionne
Durée estimée : 2-3 minutes
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import shutil
import time
from pathlib import Path
import random

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
    # Vérifier si GPU disponible (YOLO utilise PyTorch)
    try:
        import torch
        GPU_AVAILABLE = torch.cuda.is_available()
    except:
        GPU_AVAILABLE = False
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    GPU_AVAILABLE = False
    print("❌ Ultralytics non disponible. Installez avec: pip install ultralytics")

from utils.config import *
from utils.helpers import *


def create_small_dataset(source_path: str, dest_path: str, num_train: int = 200, num_val: int = 50):
    """
    Crée un petit dataset pour test rapide
    
    Args:
        source_path: Chemin vers RDD_SPLIT complet
        dest_path: Chemin où créer le petit dataset
        num_train: Nombre d'images train
        num_val: Nombre d'images validation
    """
    print(f"\n📦 Création d'un petit dataset pour test rapide...")
    print(f"  - Train: {num_train} images")
    print(f"  - Val: {num_val} images")
    
    # Créer la structure
    for split in ['train', 'val']:
        os.makedirs(os.path.join(dest_path, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(dest_path, split, 'labels'), exist_ok=True)
    
    # Copier un subset d'images train
    train_images_src = Path(source_path) / 'train' / 'images'
    train_labels_src = Path(source_path) / 'train' / 'labels'
    
    all_train_images = list(train_images_src.glob('*.jpg'))
    selected_train = random.sample(all_train_images, min(num_train, len(all_train_images)))
    
    print(f"\n⏳ Copie de {len(selected_train)} images train...")
    for img_path in selected_train:
        # Copier l'image
        dest_img = Path(dest_path) / 'train' / 'images' / img_path.name
        shutil.copy2(img_path, dest_img)
        
        # Copier le label
        label_name = img_path.stem + '.txt'
        src_label = train_labels_src / label_name
        if src_label.exists():
            dest_label = Path(dest_path) / 'train' / 'labels' / label_name
            shutil.copy2(src_label, dest_label)
    
    # Copier un subset d'images val
    val_images_src = Path(source_path) / 'val' / 'images'
    val_labels_src = Path(source_path) / 'val' / 'labels'
    
    all_val_images = list(val_images_src.glob('*.jpg'))
    selected_val = random.sample(all_val_images, min(num_val, len(all_val_images)))
    
    print(f"⏳ Copie de {len(selected_val)} images validation...")
    for img_path in selected_val:
        # Copier l'image
        dest_img = Path(dest_path) / 'val' / 'images' / img_path.name
        shutil.copy2(img_path, dest_img)
        
        # Copier le label
        label_name = img_path.stem + '.txt'
        src_label = val_labels_src / label_name
        if src_label.exists():
            dest_label = Path(dest_path) / 'val' / 'labels' / label_name
            shutil.copy2(src_label, dest_label)
    
    print(f"✅ Petit dataset créé dans: {dest_path}")
    
    return dest_path


def create_yaml_config(data_path: str, yaml_path: str):
    """
    Crée le fichier YAML de configuration pour YOLO
    """
    import yaml
    
    data_config = {
        'path': str(Path(data_path).absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'nc': 4,
        'names': {
            0: 'Longitudinal',
            1: 'Transverse',
            2: 'Crocodile',
            3: 'Pothole'
        }
    }
    
    with open(yaml_path, 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False, sort_keys=False)
    
    print(f"✅ Fichier YAML créé: {yaml_path}")
    
    return yaml_path


def ultra_fast_test(data_path: str):
    """
    Test ULTRA-RAPIDE YOLO avec seulement 200 images et 2 epochs
    """
    if not ULTRALYTICS_AVAILABLE:
        raise ImportError("Ultralytics n'est pas installé. Installez avec: pip install ultralytics")
    
    print("\n" + "=" * 100)
    print("⚡ TEST ULTRA-RAPIDE YOLO (200 IMAGES, 2 EPOCHS)")
    print("=" * 100)
    print("\n🎯 Objectif : Vérifier que l'entraînement fonctionne")
    print("⏱️  Durée estimée : 2-3 minutes")
    print("=" * 100)
    
    # Seeds
    set_seeds(RANDOM_SEED)
    
    # Créer un dossier temporaire pour le petit dataset
    temp_dataset_path = os.path.join(os.path.dirname(data_path), 'RDD_SPLIT_SMALL_TEMP')
    
    # Supprimer s'il existe déjà
    if os.path.exists(temp_dataset_path):
        print(f"\n🗑️  Suppression de l'ancien dataset temporaire...")
        shutil.rmtree(temp_dataset_path)
    
    # ========================================================================
    # 1. CRÉATION DU PETIT DATASET
    # ========================================================================
    print("\n📦 PHASE 1: Préparation des données")
    print("-" * 100)
    
    small_dataset = create_small_dataset(
        source_path=data_path,
        dest_path=temp_dataset_path,
        num_train=200,
        num_val=50
    )
    
    # Créer le fichier YAML
    yaml_path = os.path.join(temp_dataset_path, 'data.yaml')
    create_yaml_config(temp_dataset_path, yaml_path)
    
    # ========================================================================
    # 2. CHARGEMENT DU MODÈLE
    # ========================================================================
    print("\n🏗️  PHASE 2: Chargement du modèle YOLO")
    print("-" * 100)
    
    model = YOLO('yolov8n-seg.pt')
    
    print("✅ Modèle YOLOv8n-seg chargé (pré-entraîné)")
    
    # ========================================================================
    # 3. ENTRAÎNEMENT
    # ========================================================================
    print("\n🚀 PHASE 3: Entraînement du modèle")
    print("-" * 100)
    print("⏳ Ceci va prendre 2-3 minutes...")
    print("-" * 100)
    
    start_time = time.time()
    
    # Créer un dossier temporaire pour les résultats
    temp_results = os.path.join(os.path.dirname(data_path), 'yolo_temp_results')
    
    # Déterminer le device
    device = '0' if GPU_AVAILABLE else 'cpu'
    print(f"🖥️  Device utilisé: {'GPU' if GPU_AVAILABLE else 'CPU'}")
    
    # Entraîner le modèle (mode ultra-rapide)
    results = model.train(
        data=yaml_path,
        epochs=2,  # Seulement 2 epochs
        batch=8,   # Petit batch
        imgsz=640,
        lr0=0.01,  # Learning rate élevé pour test rapide
        patience=10,
        project=temp_results,
        name='ultra_fast_test',
        save=False,  # Ne pas sauvegarder pour gagner du temps
        plots=False,  # Pas de plots pour gagner du temps
        verbose=True,
        device=device,  # GPU si disponible, sinon CPU
        # Désactiver les augmentations lourdes
        mosaic=0.0,
        mixup=0.0,
        degrees=0.0,
        translate=0.0,
        scale=0.0,
        fliplr=0.5,
        # Autres paramètres
        amp=False,  # Désactiver AMP pour test rapide
        val=True
    )
    
    training_time = time.time() - start_time
    
    print("\n" + "=" * 100)
    print("✅ TEST TERMINÉ!")
    print("=" * 100)
    print(f"⏱️  Temps total: {format_time(training_time)}")
    
    # ========================================================================
    # 4. RÉSULTATS
    # ========================================================================
    print("\n📊 PHASE 4: Résultats")
    print("-" * 100)
    
    # Validation rapide
    val_results = model.val(
        data=yaml_path,
        split='val',
        batch=8,
        verbose=False
    )
    
    print("\n📊 Métriques finales:")
    print("-" * 100)
    print(f"  Box mAP50        : {val_results.box.map50:.4f}")
    print(f"  Box mAP50-95     : {val_results.box.map:.4f}")
    print(f"  Mask mAP50       : {val_results.seg.map50:.4f}")
    print(f"  Mask mAP50-95    : {val_results.seg.map:.4f}")
    print("-" * 100)
    
    # ========================================================================
    # 5. NETTOYAGE
    # ========================================================================
    print("\n🗑️  PHASE 5: Nettoyage")
    print("-" * 100)
    
    # Supprimer le dataset temporaire
    if os.path.exists(temp_dataset_path):
        shutil.rmtree(temp_dataset_path)
        print(f"✅ Dataset temporaire supprimé")
    
    # Supprimer les résultats temporaires
    if os.path.exists(temp_results):
        shutil.rmtree(temp_results)
        print(f"✅ Résultats temporaires supprimés")
    
    # ========================================================================
    # 6. CONCLUSION
    # ========================================================================
    print("\n" + "=" * 100)
    print("🎉 CONCLUSION DU TEST")
    print("=" * 100)
    
    if val_results.seg.map50 > 0.1:
        print("✅ Le modèle fonctionne correctement!")
        print(f"✅ Mask mAP50 = {val_results.seg.map50:.4f}")
        print("\n💡 Tu peux maintenant lancer l'entraînement complet:")
        print("   python training/train_yolo.py")
    else:
        print("⚠️  Les métriques sont basses (normal pour 2 epochs)")
        print("⚠️  Mais le code fonctionne correctement!")
        print("\n💡 Pour de meilleurs résultats, lance l'entraînement complet:")
        print("   python training/train_yolo.py")
    
    print("=" * 100)
    
    return model, results


if __name__ == "__main__":
    # Chemin du dataset
    DATA_PATH = "C:/Users/DELL/Desktop/dataset/RDD_SPLIT"
    
    # Vérifier que le chemin existe
    if not os.path.exists(DATA_PATH):
        print("\n" + "=" * 100)
        print("❌ ERREUR: Le chemin du dataset n'existe pas!")
        print("=" * 100)
        print(f"\nChemin spécifié: {DATA_PATH}")
        print("\nVeuillez modifier la variable DATA_PATH dans ce script.")
        print("=" * 100)
    else:
        # Lancer le test ultra-rapide
        model, results = ultra_fast_test(DATA_PATH)