import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import random
import os
import pickle
from streamlit_drawable_canvas import st_canvas
import matplotlib.pyplot as plt

# Sayfa ayarları
st.set_page_config(
    page_title="🎨 Almanca Çizim Oyunu",
    page_icon="🎨",
    layout="wide"
)

class GermanDrawingGame:
    def __init__(self):
        print("GermanDrawingGame başlatılıyor...")
        print("load_model_and_classes çağrılıyor...")
        result = self.load_model_and_classes()
        print(f"load_model_and_classes sonucu: {type(result)}")
    
        self.model, self.classes, self.german_translations = result
    
        print(f"Model: {type(self.model)}")
        print(f"Classes: {type(self.classes)} - {len(self.classes) if hasattr(self.classes, '__len__') else 'No len'}")
        print(f"German translations: {type(self.german_translations)}")
        
    def load_model_and_classes(self):
        print("Model yükleniyor...")
    
        # Model yükleme
        model_path = os.path.join("model", "ensemble_model_2_best.h5")
        print(f"Model path: {model_path}")
    
        if not os.path.exists(model_path):
            print("Model dosyası bulunamadı!")
            st.error(f"Model dosyası bulunamadı: {model_path}")
            st.stop()

        def focal_loss(gamma=2.0, alpha=0.25):
            def focal_loss_fixed(y_true, y_pred):
                import tensorflow as tf
                epsilon = tf.keras.backend.epsilon()
                y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        
                ce = -y_true * tf.math.log(y_pred)
                weight = alpha * y_true * tf.pow((1 - y_pred), gamma)
                fl = weight * ce
        
                return tf.reduce_mean(tf.reduce_sum(fl, axis=1))
    
            return focal_loss_fixed 
        
        print("TensorFlow model yükleniyor...")
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={'focal_loss_fixed': focal_loss()}   # ✅ ekstra () kaldırıldı
        )
        print("Model yüklendi!")
    
        # Sınıfları pickle dosyasından yükle
        print("Kategoriler yükleniyor...")
        pickle_path = os.path.join("model", "category_names.pkl")
    
        with open(pickle_path, 'rb') as f:
            classes = pickle.load(f)
    
        print(f"Kategoriler yüklendi: {len(classes)}")
    
        # Almanca çevirileri
        german_translations = {
            'airplane': 'das Flugzeug', 'angel': 'der Engel', 'apple': 'der Apfel'
            # ... (kısaltılmış)
        }
    
        print("Her şey hazır, return ediliyor...")
        return model, classes, german_translations
    
    def preprocess_drawing(self, image_data):
        """Çizimi model için hazırla"""
        if image_data is None:
            return None
            
        # RGBA veriyi Image nesnesine dönüştür
        img = Image.fromarray(image_data.astype('uint8'), 'RGBA')
        
        # Yeni beyaz bir arka plan oluştur
        new_img = Image.new("RGB", img.size, (255, 255, 255))
        
        # Çizimi yeni beyaz arka plana yapıştır
        new_img.paste(img, (0, 0), img)
        
        # Gri tonlamaya çevir ve 28x28'e boyutlandır
        img_gray = new_img.convert('L')
        img_resized = img_gray.resize((28, 28), Image.Resampling.LANCZOS)
        
        # Numpy array'e dönüştür
        img_array = np.array(img_resized)
        
        # Çizgileri beyaza, arka planı siyaha çevir (Quick, Draw! formatına uygun)
        img_array = 255.0 - img_array
        
        # Normalize et (0-1 arasına getir)
        img_array = img_array.astype('float32') / 255.0
        
        # Model formatına getir (1, 28, 28, 1)
        img_array = img_array.reshape(1, 28, 28, 1)
        
        return img_array

    def predict_drawing(self, processed_image):
        """Çizimi tahmin et"""
        if self.model is None or processed_image is None:
            return None, None
            
        predictions = self.model.predict(processed_image, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_idx]
        
        if predicted_class_idx < len(self.classes):
            predicted_class = self.classes[predicted_class_idx]
        else:
            predicted_class = "unknown"
        
        return predicted_class, confidence

# Ana uygulama
def main():
    st.title("🎨 Almanca Çizim Oyunu")
    st.markdown("---")
    
    # Session state başlatma
    if 'game' not in st.session_state:
        try:
            st.session_state.game = GermanDrawingGame()
            st.session_state.current_word = None
            st.session_state.score = 0
            st.session_state.round_count = 0
            st.session_state.game_started = False
            st.session_state.canvas_key = "0"   # 🔑 string başlangıç
        except Exception as e:
            st.error(f"Oyun başlatılırken hata: {e}")
            return
    
    # Sidebar - Oyun kontrolleri
    with st.sidebar:
        st.header("🎮 Oyun Kontrolü")
        
        if st.button("🆕 Yeni Oyun Başlat"):
            st.session_state.score = 0
            st.session_state.round_count = 0
            st.session_state.game_started = True
            st.session_state.current_word = random.choice(st.session_state.game.classes)
            st.session_state.canvas_key = str(random.randint(0, 10000))  # ✅ string key
            st.rerun()
        
        if st.button("🔄 Yeni Kelime"):
            if st.session_state.game_started:
                st.session_state.current_word = random.choice(st.session_state.game.classes)
                st.session_state.canvas_key = str(random.randint(0, 10000))  # ✅ string key
                st.rerun()
        
        st.markdown("---")
        st.metric("🏆 Skor", st.session_state.score)
        st.metric("🔢 Round", st.session_state.round_count)
        
        st.header("🖌️ Çizim Ayarları")
        drawing_mode = st.selectbox("Mod:", ("freedraw", "transform", "point"), index=0)
        stroke_width = st.slider("Çizgi Kalınlığı:", 1, 25, 3)
        stroke_color = st.color_picker("Çizgi Rengi:", "#000000")

    # Ana oyun alanı
    if not st.session_state.game_started:
        st.info("🎯 Oyunu başlatmak için yan menüden 'Yeni Oyun Başlat' butonuna tıklayın!")
        st.markdown("""
        ### 🎮 Nasıl Oynanır?
        1. **🆕 Yeni Oyun Başlat** butonuna tıklayın
        2. **🤖 Robot size bir Almanca kelime verecek**
        3. **🎨 Bu kelimeyi canvas üzerine çizin**
        4. **🤖 Robot çiziminizi tahmin etmeye çalışacak**
        5. **🏆 Doğru tahmin edilirse puan kazanırsınız!**
        """)
        return
    
    if st.session_state.current_word:
        st.markdown("### 🤖 Robot der ki:")
        german_word = st.session_state.game.german_translations.get(
            st.session_state.current_word, 
            st.session_state.current_word
        )
        st.success(f"**Çiz bana: {german_word}** ({st.session_state.current_word})")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 🎨 Çizim Alanı")
            
            canvas_result = st_canvas(
                fill_color="rgba(255, 255, 255, 0.3)",
                stroke_width=stroke_width,
                stroke_color=stroke_color,
                background_color="white",
                height=400,
                width=400,
                drawing_mode=drawing_mode,
                key=st.session_state.canvas_key,   # ✅ artık string
            )
            
            if st.button("🔮 Tahmin Et!", type="primary"):
                if canvas_result.image_data is not None:
                    # Çizimi işle
                    processed_img = st.session_state.game.preprocess_drawing(canvas_result.image_data)
                    
                    if processed_img is not None:
                        # Tahmin yap
                        prediction, confidence = st.session_state.game.predict_drawing(processed_img)
                        
                        st.markdown("### 🤖 Robot'un Tahmini:")
                        
                        # Güven skoruna bakarak değerlendirme
                        if prediction == st.session_state.current_word and confidence > 0.3:
                            st.success(f"🎉 **Harika! Bu bir {st.session_state.game.german_translations.get(prediction, prediction)}!**")
                            st.success(f"Güven: {confidence:.2%}")
                            st.session_state.score += 10
                            st.session_state.round_count += 1
                            st.balloons()
                        else:
                            german_pred = st.session_state.game.german_translations.get(prediction, prediction)
                            st.error(f"🤔 **Bu bana {german_pred} gibi görünüyor.**")
                            st.info(f"Güven: {confidence:.2%}")
                            st.info("Tekrar deneyin veya yeni kelime alın!")
                        
                        # İşlenmiş görüntüyü göster
                        with col2:
                            st.markdown("### 🔍 İşlenmiş Görüntü")
                            fig, ax = plt.subplots(figsize=(4, 4))
                            ax.imshow(processed_img[0, :, :, 0], cmap='gray')
                            ax.axis('off')
                            st.pyplot(fig)
                            plt.close(fig)
                else:
                    st.warning("Önce bir şeyler çizin!")
        
        with col2:
            st.markdown("### 💡 İpuçları")
            st.info(f"""
            **Kelime:** {german_word}
            
            🎯 **İpuçları:**
            - Kalın çizgilerle çizin
            - Ana şekli vurgulamaya odaklanın  
            - Çok detaya girmeyin
            - Merkeze çizin
            """)
    
    # Alt bilgi
    st.markdown("---")
    st.markdown("**🎓 Bu oyun Almanca kelime öğrenmenize yardımcı olur!**")

if __name__ == "__main__":
    main()
