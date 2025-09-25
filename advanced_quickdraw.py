import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
import pickle
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Seçilen 50 kategori
SELECTED_CATEGORIES = [
    'airplane', 'angel', 'apple', 'banana', 'bear', 'bicycle', 'bird', 'book', 
    'bus', 'cake', 'camel', 'car', 'cat', 'chair', 'clock', 'cloud', 'computer', 
    'cow', 'dog', 'duck', 'ear', 'elephant', 'eye', 'face', 'finger', 'fish', 
    'flower', 'foot', 'fork', 'frog', 'guitar', 'hammer', 'hand', 'hat', 'house', 
    'ladder', 'moon', 'mountain', 'mushroom', 'nail', 'pencil', 'piano', 'pillow', 
    'pizza', 'rainbow', 'shoe', 'star', 'strawberry', 'sun', 'table'
]

class AdvancedQuickDrawDataLoader:
    def __init__(self, data_path, max_samples_per_class=2500):
        self.data_path = data_path
        self.max_samples = max_samples_per_class
        self.categories = SELECTED_CATEGORIES
        
    def load_and_preprocess_data(self):
        """Gelişmiş veri yükleme ve ön işleme"""
        X_data = []
        y_data = []
        class_samples = {}
        
        print("Gelişmiş veri yükleniyor...")
        for idx, category in enumerate(self.categories):
            file_path = os.path.join(self.data_path, f"full_numpy_bitmap_{category}.npy")
            
            if not os.path.exists(file_path):
                print(f"UYARI: {file_path} dosyası bulunamadı!")
                continue
                
            print(f"Yükleniyor: {category} ({idx+1}/{len(self.categories)})")
            
            # Veriyi yükle
            data = np.load(file_path)
            
            # Maksimum örnek sayısını sınırla
            if len(data) > self.max_samples:
                indices = np.random.choice(len(data), self.max_samples, replace=False)
                data = data[indices]
            
            # Veriyi normalize et ve reshape et
            data = data.astype('float32') / 255.0
            data = data.reshape(-1, 28, 28, 1)
            
            # Edge detection ön işleme (opsiyonel)
            data_enhanced = self.enhance_images(data)
            
            X_data.append(data_enhanced)
            y_data.extend([idx] * len(data_enhanced))
            class_samples[category] = len(data_enhanced)
            
            print(f"  -> {len(data_enhanced)} örnek yüklendi")
        
        X_data = np.vstack(X_data)
        y_data = np.array(y_data)
        
        print(f"\nToplam: {len(X_data)} örnek, {len(self.categories)} sınıf")
        print("\nSınıf dağılımı:")
        for cat, count in class_samples.items():
            print(f"  {cat}: {count}")
            
        return X_data, y_data, class_samples
    
    def enhance_images(self, images):
        """Görüntü geliştirme teknikleri"""
        # Basit edge enhancement - sadece orijinal görüntüleri döndür
        # (GPU bellek tasarrufu için)
        return images

def create_data_generators(X_train, y_train, X_val, y_val, batch_size=64):
    """Gelişmiş veri artırma generatörleri"""
    # Eğitim veri artırma
    train_datagen = ImageDataGenerator(
        rotation_range=15,          # 15 derece döndürme
        width_shift_range=0.1,      # Yatay kaydırma
        height_shift_range=0.1,     # Dikey kaydırma
        zoom_range=0.1,             # Zoom
        shear_range=0.1,            # Eğme
        fill_mode='constant',       # Boş alanları 0 ile doldur
        cval=0.0
    )
    
    # Validasyon veri artırma (sadece normalizasyon)
    val_datagen = ImageDataGenerator()
    
    train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)
    val_generator = val_datagen.flow(X_val, y_val, batch_size=batch_size)
    
    return train_generator, val_generator

def create_advanced_model(num_classes, input_shape=(28, 28, 1)):
    """Gelişmiş ResNet-inspired CNN model"""
    inputs = layers.Input(shape=input_shape)
    
    # İlk konvolüsyon bloğu
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    # Residual Block 1
    residual1 = x
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    # Residual connection
    residual1 = layers.Conv2D(64, (1, 1), padding='same')(residual1)  # Dimension match
    x = layers.Add()([x, residual1])
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    # Residual Block 2
    residual2 = x
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    # Residual connection
    residual2 = layers.Conv2D(128, (1, 1), padding='same')(residual2)
    x = layers.Add()([x, residual2])
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    # Attention Mechanism (Basit)
    attention = layers.GlobalAveragePooling2D()(x)
    attention = layers.Dense(128, activation='relu')(attention)
    attention = layers.Dense(128, activation='sigmoid')(attention)
    attention = layers.Reshape((1, 1, 128))(attention)
    x = layers.Multiply()([x, attention])
    
    # Global pooling ve classifier
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    return model

def focal_loss(gamma=2.0, alpha=0.25):
    """Focal Loss - zor örneklere odaklanır"""
    def focal_loss_fixed(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        
        ce = -y_true * tf.math.log(y_pred)
        weight = alpha * y_true * tf.pow((1 - y_pred), gamma)
        fl = weight * ce
        
        return tf.reduce_mean(tf.reduce_sum(fl, axis=1))
    
    return focal_loss_fixed

def create_advanced_callbacks(model_name):
    """Gelişmiş callback'ler"""
    callbacks_list = [
        # Model checkpoint
        callbacks.ModelCheckpoint(
            f'{model_name}_best.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1,
            mode='max'
        ),
        
        # Early stopping
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=8,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Learning rate scheduler
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=4,
            min_lr=1e-7,
            verbose=1
        ),
        
        # Cosine annealing
        callbacks.LearningRateScheduler(
            lambda epoch: 0.001 * (np.cos(epoch / 50 * np.pi) * 0.5 + 0.5)
        ),
        
        # CSV logger
        callbacks.CSVLogger(f'{model_name}_training.csv')
    ]
    
    return callbacks_list

def plot_comprehensive_results(history, class_samples, model_name):
    """Kapsamlı sonuç görselleştirmesi"""
    fig = plt.figure(figsize=(20, 12))
    
    # Eğitim geçmişi
    ax1 = plt.subplot(2, 3, 1)
    plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu', linewidth=2)
    plt.plot(history.history['val_accuracy'], label='Validasyon Doğruluğu', linewidth=2)
    plt.title('Model Doğruluğu', fontsize=14)
    plt.xlabel('Epoch')
    plt.ylabel('Doğruluk')
    plt.legend()
    plt.grid(True)
    
    ax2 = plt.subplot(2, 3, 2)
    plt.plot(history.history['loss'], label='Eğitim Kaybı', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validasyon Kaybı', linewidth=2)
    plt.title('Model Kaybı', fontsize=14)
    plt.xlabel('Epoch')
    plt.ylabel('Kayıp')
    plt.legend()
    plt.grid(True)
    
    # Learning rate
    if 'lr' in history.history:
        ax3 = plt.subplot(2, 3, 3)
        plt.plot(history.history['lr'], linewidth=2)
        plt.title('Learning Rate', fontsize=14)
        plt.xlabel('Epoch')
        plt.ylabel('LR')
        plt.yscale('log')
        plt.grid(True)
    
    # Sınıf dağılımı
    ax4 = plt.subplot(2, 3, 4)
    categories = list(class_samples.keys())
    counts = list(class_samples.values())
    plt.bar(range(len(categories)), counts)
    plt.title('Sınıf Dağılımı', fontsize=14)
    plt.xlabel('Kategori')
    plt.ylabel('Örnek Sayısı')
    plt.xticks(range(0, len(categories), 5), [categories[i] for i in range(0, len(categories), 5)], rotation=45)
    
    # En iyi epoch bilgileri
    ax5 = plt.subplot(2, 3, 5)
    best_epoch = np.argmax(history.history['val_accuracy'])
    best_acc = max(history.history['val_accuracy'])
    best_loss = history.history['val_loss'][best_epoch]
    
    info_text = f"""
    En İyi Epoch: {best_epoch + 1}
    En İyi Val Acc: {best_acc:.4f}
    Val Loss: {best_loss:.4f}
    
    Final Train Acc: {history.history['accuracy'][-1]:.4f}
    Final Val Acc: {history.history['val_accuracy'][-1]:.4f}
    """
    
    plt.text(0.1, 0.5, info_text, fontsize=12, verticalalignment='center')
    plt.axis('off')
    plt.title('Eğitim Özeti', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(f'{model_name}_comprehensive_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def ensemble_training(X, y, class_samples, n_models=3):
    """Ensemble model eğitimi"""
    models_list = []
    histories = []
    
    kfold = StratifiedKFold(n_splits=n_models, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
        print(f"\n{'='*50}")
        print(f"ENSEMBLE MODEL {fold + 1}/{n_models}")
        print(f"{'='*50}")
        
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        # One-hot encoding
        y_train_cat = tf.keras.utils.to_categorical(y_train_fold, len(SELECTED_CATEGORIES))
        y_val_cat = tf.keras.utils.to_categorical(y_val_fold, len(SELECTED_CATEGORIES))
        
        # Model oluştur
        model = create_advanced_model(len(SELECTED_CATEGORIES))
        
        # Class weights hesapla
        class_weights = compute_class_weight(
            'balanced', 
            classes=np.unique(y_train_fold), 
            y=y_train_fold
        )
        class_weight_dict = dict(enumerate(class_weights))
        
        # Compile
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=focal_loss(gamma=2.0, alpha=0.25),
            metrics=['accuracy']
        )
        
        # Veri generatörleri
        train_gen, val_gen = create_data_generators(
            X_train_fold, y_train_cat, X_val_fold, y_val_cat, batch_size=64
        )
        
        # Callbacks
        callbacks_list = create_advanced_callbacks(f'ensemble_model_{fold+1}')
        
        # Eğitim
        history = model.fit(
            train_gen,
            epochs=40,
            validation_data=val_gen,
            callbacks=callbacks_list,
            class_weight=class_weight_dict,
            verbose=1
        )
        
        models_list.append(model)
        histories.append(history)
    
    return models_list, histories

def main():
    print("🚀 GELİŞMİŞ QUICKDRAW CNN EĞİTİMİ BAŞLIYOR 🚀")
    
    # GPU kontrolü
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✅ GPU bulundu: {len(gpus)} adet")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("⚠️  GPU bulunamadı, CPU ile eğitim yapılacak")
    
    # Veri yolu
    DATA_PATH = "C:/projects/draw4german_game/data"
    
    # Veri yükleme
    loader = AdvancedQuickDrawDataLoader(DATA_PATH, max_samples_per_class=2500)
    X, y, class_samples = loader.load_and_preprocess_data()
    
    # Veri ayrımı
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42
    )
    
    print(f"\n📊 VERİ ÖZETİ:")
    print(f"Eğitim seti: {len(X_train):,}")
    print(f"Test seti: {len(X_test):,}")
    print(f"Toplam sınıf: {len(SELECTED_CATEGORIES)}")
    
    # Ensemble eğitimi
    print(f"\n🎯 ENSEMBLE EĞİTİMİ BAŞLIYOR...")
    ensemble_models, ensemble_histories = ensemble_training(X_train, y_train, class_samples)
    
    # Ensemble değerlendirme
    print(f"\n📈 ENSEMBLE DEĞERLENDİRME:")
    ensemble_predictions = []
    
    for i, model in enumerate(ensemble_models):
        y_test_cat = tf.keras.utils.to_categorical(y_test, len(SELECTED_CATEGORIES))
        pred = model.predict(X_test, verbose=0)
        ensemble_predictions.append(pred)
        
        test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
        print(f"Model {i+1} Test Doğruluğu: {test_acc:.4f}")
    
    # Ensemble ortalama
    ensemble_pred = np.mean(ensemble_predictions, axis=0)
    ensemble_acc = np.mean(np.argmax(ensemble_pred, axis=1) == y_test)
    
    print(f"\n🏆 ENSEMBLE SONUÇLARI:")
    print(f"🎯 Ensemble Doğruluğu: {ensemble_acc:.4f}")
    print(f"📈 En iyi tekil model: {max([max(h.history['val_accuracy']) for h in ensemble_histories]):.4f}")
    print(f"🚀 İyileştirme: +{(ensemble_acc - max([max(h.history['val_accuracy']) for h in ensemble_histories])):.4f}")
    
    # Sonuçları kaydet
    results = {
        'ensemble_accuracy': ensemble_acc,
        'individual_accuracies': [max(h.history['val_accuracy']) for h in ensemble_histories],
        'category_names': SELECTED_CATEGORIES,
        'class_samples': class_samples
    }
    
    with open('advanced_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    # En iyi modelin sonuçlarını görselleştir
    best_model_idx = np.argmax([max(h.history['val_accuracy']) for h in ensemble_histories])
    plot_comprehensive_results(
        ensemble_histories[best_model_idx], 
        class_samples, 
        f'advanced_model_best'
    )
    
    print(f"\n✅ EĞİTİM TAMAMLANDI!")
    print(f"📁 Kaydedilen dosyalar:")
    print(f"  - ensemble_model_1_best.h5, ensemble_model_2_best.h5, ...")
    print(f"  - advanced_results.pkl")
    print(f"  - advanced_model_best_comprehensive_results.png")

if __name__ == "__main__":
    main()