"""
Script d'Entraînement Rapide avec 1000 Images
Pour tester les 3 architectures rapidement
"""

import sys
import os

# Ajouter le répertoire parent au path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from preprocessing.data_loader import RDD2022DataLoader
from preprocessing.preprocessing import ImagePreprocessor
from models.unet_scratch import create_unet_model
from models.hybrid_model import create_hybrid_model
from models.model_utils import DiceCoefficient, IoUMetric
from training.callbacks import create_callbacks
from utils.config import *
from utils.helpers import set_seeds

import tensorflow as tf


def train_unet_1000(data_path, epochs=10):
    """
    Entraîne U-Net sur 1000 images
    """
    print("\n" + "=" * 100)
    print("⚡ ENTRAÎNEMENT U-NET - 1000 IMAGES")
    print("=" * 100)
    
    # Configuration
    set_seeds(RANDOM_SEED)
    
    # 1. Charger les données
    print("\n📦 PHASE 1: Chargement des données (1000 images)")
    print("-" * 100)
    
    train_loader = RDD2022DataLoader(data_path, split='train')
    val_loader = RDD2022DataLoader(data_path, split='val')
    
    # Prendre 1000 images train, 200 val
    num_train = min(1000, len(train_loader))
    num_val = min(200, len(val_loader))
    
    print(f"✅ Train: {num_train} images")
    print(f"✅ Val: {num_val} images")
    
    # Charger les images
    preprocessor = ImagePreprocessor(target_size=IMG_SIZE, normalize=True)
    
    train_images = []
    train_masks = []
    for i in range(num_train):
        if i % 200 == 0:
            print(f"  - Chargement train: {i}/{num_train}...")
        img, mask, _ = train_loader[i]
        train_images.append(img)
        train_masks.append(mask)
    
    val_images = []
    val_masks = []
    for i in range(num_val):
        if i % 50 == 0:
            print(f"  - Chargement val: {i}/{num_val}...")
        img, mask, _ = val_loader[i]
        val_images.append(img)
        val_masks.append(mask)
    
    # Prétraiter
    print("\n🔧 Prétraitement des données...")
    X_train, y_train = preprocessor.preprocess_batch(train_images, train_masks)
    X_val, y_val = preprocessor.preprocess_batch(val_images, val_masks)
    
    # Convertir en categorical
    y_train_cat = np.array([preprocessor.mask_to_categorical(m) for m in y_train])
    y_val_cat = np.array([preprocessor.mask_to_categorical(m) for m in y_val])
    
    print(f"✅ X_train: {X_train.shape}")
    print(f"✅ y_train: {y_train_cat.shape}")
    print(f"✅ X_val: {X_val.shape}")
    print(f"✅ y_val: {y_val_cat.shape}")
    
    # 2. Créer le modèle
    print("\n🏗️  PHASE 2: Création du modèle U-Net")
    print("-" * 100)
    
    model = create_unet_model(
        input_shape=IMG_SIZE + (IMG_CHANNELS,),
        num_classes=NUM_CLASSES + 1,
        filters_base=64,
        compile_model=True
    )
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy', DiceCoefficient(), IoUMetric()]
    )
    
    print("✅ Modèle créé et compilé")
    
    # 3. Callbacks
    print("\n📊 PHASE 3: Configuration des callbacks")
    print("-" * 100)
    
    callbacks = create_callbacks(
        model_name='unet_1000img',
        models_dir=MODELS_DIR,
        log_dir=LOGS_DIR,
        monitor='val_dice_coefficient',
        patience_early_stop=10,
        patience_reduce_lr=5
    )
    
    print(f"✅ {len(callbacks)} callbacks configurés")
    
    # 4. Entraînement
    print("\n🚀 PHASE 4: Entraînement du modèle")
    print("=" * 100)
    print(f"⏱️  Durée estimée: 10-15 minutes")
    print("=" * 100 + "\n")
    
    history = model.fit(
        X_train, y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=epochs,
        batch_size=16,
        callbacks=callbacks,
        verbose=1
    )
    
    # 5. Résultats
    print("\n" + "=" * 100)
    print("✅ ENTRAÎNEMENT U-NET TERMINÉ (1000 images)")
    print("=" * 100)
    
    final_metrics = {
        'train_loss': history.history['loss'][-1],
        'train_dice': history.history['dice_coefficient'][-1],
        'train_iou': history.history['iou'][-1],
        'val_loss': history.history['val_loss'][-1],
        'val_dice': history.history['val_dice_coefficient'][-1],
        'val_iou': history.history['val_iou'][-1]
    }
    
    print(f"\n📊 Métriques finales:")
    for key, value in final_metrics.items():
        print(f"  - {key}: {value:.4f}")
    
    return model, history


def train_hybrid_1000(data_path, epochs=10):
    """
    Entraîne le modèle Hybride sur 1000 images
    """
    print("\n" + "=" * 100)
    print("⚡ ENTRAÎNEMENT HYBRIDE - 1000 IMAGES")
    print("=" * 100)
    
    # Configuration
    set_seeds(RANDOM_SEED)
    
    # 1. Charger les données (même code que U-Net)
    print("\n📦 PHASE 1: Chargement des données (1000 images)")
    print("-" * 100)
    
    train_loader = RDD2022DataLoader(data_path, split='train')
    val_loader = RDD2022DataLoader(data_path, split='val')
    
    num_train = min(1000, len(train_loader))
    num_val = min(200, len(val_loader))
    
    print(f"✅ Train: {num_train} images")
    print(f"✅ Val: {num_val} images")
    
    # Charger et prétraiter
    preprocessor = ImagePreprocessor(target_size=IMG_SIZE, normalize=True)
    
    train_images = []
    train_masks = []
    for i in range(num_train):
        if i % 200 == 0:
            print(f"  - Chargement train: {i}/{num_train}...")
        img, mask, _ = train_loader[i]
        train_images.append(img)
        train_masks.append(mask)
    
    val_images = []
    val_masks = []
    for i in range(num_val):
        if i % 50 == 0:
            print(f"  - Chargement val: {i}/{num_val}...")
        img, mask, _ = val_loader[i]
        val_images.append(img)
        val_masks.append(mask)
    
    print("\n🔧 Prétraitement des données...")
    X_train, y_train = preprocessor.preprocess_batch(train_images, train_masks)
    X_val, y_val = preprocessor.preprocess_batch(val_images, val_masks)
    
    y_train_cat = np.array([preprocessor.mask_to_categorical(m) for m in y_train])
    y_val_cat = np.array([preprocessor.mask_to_categorical(m) for m in y_val])
    
    print(f"✅ Données prêtes")
    
    # 2. Créer le modèle
    print("\n🏗️  PHASE 2: Création du modèle Hybride")
    print("-" * 100)
    
    model = create_hybrid_model(
        input_shape=IMG_SIZE + (IMG_CHANNELS,),
        num_classes=NUM_CLASSES + 1,
        filters_base=64,
        compile_model=True
    )
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy', DiceCoefficient(), IoUMetric()]
    )
    
    print("✅ Modèle créé et compilé")
    
    # 3. Callbacks
    callbacks = create_callbacks(
        model_name='hybrid_1000img',
        models_dir=MODELS_DIR,
        log_dir=LOGS_DIR,
        monitor='val_dice_coefficient',
        patience_early_stop=10,
        patience_reduce_lr=5
    )
    
    # 4. Entraînement
    print("\n🚀 PHASE 4: Entraînement du modèle")
    print("=" * 100)
    print(f"⏱️  Durée estimée: 12-18 minutes")
    print("=" * 100 + "\n")
    
    history = model.fit(
        X_train, y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=epochs,
        batch_size=16,
        callbacks=callbacks,
        verbose=1
    )
    
    # 5. Résultats
    print("\n" + "=" * 100)
    print("✅ ENTRAÎNEMENT HYBRIDE TERMINÉ (1000 images)")
    print("=" * 100)
    
    final_metrics = {
        'train_loss': history.history['loss'][-1],
        'train_dice': history.history['dice_coefficient'][-1],
        'train_iou': history.history['iou'][-1],
        'val_loss': history.history['val_loss'][-1],
        'val_dice': history.history['val_dice_coefficient'][-1],
        'val_iou': history.history['val_iou'][-1]
    }
    
    print(f"\n📊 Métriques finales:")
    for key, value in final_metrics.items():
        print(f"  - {key}: {value:.4f}")
    
    return model, history


def train_yolo_1000(data_path):
    """
    Entraîne YOLO sur 1000 images
    """
    print("\n" + "=" * 100)
    print("⚡ ENTRAÎNEMENT YOLO - 1000 IMAGES")
    print("=" * 100)
    
    try:
        from ultralytics import YOLO
        from models.yolo_pretrained import YOLOSegmentation, create_yolo_data_yaml
        import shutil
        from pathlib import Path
        
        # Créer un dataset temporaire avec 1000 images
        print("\n📦 Création d'un dataset temporaire (1000 images)...")
        
        temp_dir = Path(data_path).parent / 'RDD_SPLIT_1000'
        
        # Supprimer l'ancien si existe
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        
        # Créer la structure
        (temp_dir / 'train' / 'images').mkdir(parents=True, exist_ok=True)
        (temp_dir / 'train' / 'labels').mkdir(parents=True, exist_ok=True)
        (temp_dir / 'val' / 'images').mkdir(parents=True, exist_ok=True)
        (temp_dir / 'val' / 'labels').mkdir(parents=True, exist_ok=True)
        
        # Copier 1000 images train
        source_train_img = Path(data_path) / 'train' / 'images'
        source_train_lbl = Path(data_path) / 'train' / 'labels'
        
        train_files = list(source_train_img.glob('*.jpg'))[:1000]
        
        print(f"  - Copie de {len(train_files)} images train...")
        for img_file in train_files:
            shutil.copy(img_file, temp_dir / 'train' / 'images')
            lbl_file = source_train_lbl / (img_file.stem + '.txt')
            if lbl_file.exists():
                shutil.copy(lbl_file, temp_dir / 'train' / 'labels')
        
        # Copier 200 images val
        source_val_img = Path(data_path) / 'val' / 'images'
        source_val_lbl = Path(data_path) / 'val' / 'labels'
        
        val_files = list(source_val_img.glob('*.jpg'))[:200]
        
        print(f"  - Copie de {len(val_files)} images validation...")
        for img_file in val_files:
            shutil.copy(img_file, temp_dir / 'val' / 'images')
            lbl_file = source_val_lbl / (img_file.stem + '.txt')
            if lbl_file.exists():
                shutil.copy(lbl_file, temp_dir / 'val' / 'labels')
        
        print(f"✅ Dataset temporaire créé: {temp_dir}")
        
        # Créer le fichier YAML
        yaml_path = temp_dir / 'data.yaml'
        create_yolo_data_yaml(
            train_path=str(temp_dir / 'train'),
            val_path=str(temp_dir / 'val'),
            output_path=str(yaml_path)
        )
        
        # Créer le modèle YOLO
        print("\n🏗️  Chargement du modèle YOLO...")
        yolo = YOLOSegmentation(
            model_name='yolov8n-seg.pt',
            num_classes=NUM_CLASSES,
            img_size=640
        )
        yolo.load_pretrained()
        
        # Entraîner
        print("\n🚀 Entraînement YOLO (1000 images, 10 epochs)...")
        print("⏱️  Durée estimée: 8-12 minutes")
        print("=" * 100 + "\n")
        
        results = yolo.model.train(
            data=str(yaml_path),
            epochs=10,
            batch=8,
            imgsz=640,
            project=str(Path(data_path).parent / 'yolo_temp_results'),
            name='yolo_1000img',
            patience=10,
            save=True,
            plots=True,
            verbose=True
        )
        
        print("\n✅ ENTRAÎNEMENT YOLO TERMINÉ (1000 images)")
        
        # Nettoyage
        print("\n🗑️  Suppression du dataset temporaire...")
        shutil.rmtree(temp_dir)
        
        return yolo, results
        
    except ImportError:
        print("❌ Ultralytics non disponible!")
        print("   Installez avec: pip install ultralytics --break-system-packages")
        return None, None


if __name__ == "__main__":
    # Chemin vers le dataset
    DATA_PATH = "C:/Users/DELL/Desktop/dataset/RDD_SPLIT"
    
    # Vérifier que le chemin existe
    if not os.path.exists(DATA_PATH):
        print(f"❌ ERREUR: Le dataset n'existe pas à {DATA_PATH}")
        sys.exit(1)
    
    # Menu
    print("\n" + "=" * 100)
    print("⚡ ENTRAÎNEMENT RAPIDE - 1000 IMAGES")
    print("=" * 100)
    print("\nQuel modèle voulez-vous entraîner ?")
    print("  1. U-Net (10 epochs, ~12 min)")
    print("  2. Hybride (10 epochs, ~15 min)")
    print("  3. YOLO (10 epochs, ~10 min)")
    print("  4. TOUS (30 epochs total, ~40 min)")
    print()
    
    choice = input("Votre choix (1-4): ").strip()
    
    if choice == '1':
        model, history = train_unet_1000(DATA_PATH, epochs=10)
        
    elif choice == '2':
        model, history = train_hybrid_1000(DATA_PATH, epochs=10)
        
    elif choice == '3':
        model, results = train_yolo_1000(DATA_PATH)
        
    elif choice == '4':
        print("\n🚀 Entraînement des 3 modèles...")
        
        # U-Net
        print("\n" + "🔹" * 50)
        print("1/3 - U-NET")
        print("🔹" * 50)
        unet_model, unet_history = train_unet_1000(DATA_PATH, epochs=10)
        
        # Hybride
        print("\n" + "🔹" * 50)
        print("2/3 - HYBRIDE")
        print("🔹" * 50)
        hybrid_model, hybrid_history = train_hybrid_1000(DATA_PATH, epochs=10)
        
        # YOLO
        print("\n" + "🔹" * 50)
        print("3/3 - YOLO")
        print("🔹" * 50)
        yolo_model, yolo_results = train_yolo_1000(DATA_PATH)
        
        print("\n" + "=" * 100)
        print("🎉 TOUS LES ENTRAÎNEMENTS SONT TERMINÉS!")
        print("=" * 100)
        
    else:
        print("❌ Choix invalide!")
