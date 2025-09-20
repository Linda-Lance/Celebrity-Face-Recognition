# 🎭 Celebrity Face Recognition using CNN

This project implements a **Convolutional Neural Network (CNN)** for recognizing celebrity faces.  
It is trained on a custom dataset (Mammootty vs Mohanlal), and the final trained model is saved as `face_model.h5`.  
The system can classify input images into their respective celebrity category.

---

## 📂 Dataset
**Kaggle**:  https://www.kaggle.com/datasets/fillerink/mohanlal-mammooty-images
  
---

## 🧠 Model
- **Architecture**: Convolutional Neural Network (CNN) built using TensorFlow/Keras.  
- **Layers**: Convolutional layers, MaxPooling, Dense layers.  
- **Output**: Softmax activation for multi-class classification.  
- **Trained Model**: Saved as `face_model.h5`.

---

## ✨ Features
- Celebrity face detection and classification.  
- Preprocessing of dataset (resizing, normalization).  
- Model training and evaluation.  
- Visualizations of accuracy and loss.  
- Jupyter Notebook implementation for easy experimentation.  

---

## 📊 Results
- **Training Accuracy (Final): ~74%**  
- **Epochs Trained: 50**  
<img width="634" height="429" alt="Live_mammootty" src="https://github.com/user-attachments/assets/b2c21627-72d8-488b-8b58-ee49bb5c95fc" />
<img width="638" height="379" alt="Live_mohanlal" src="https://github.com/user-attachments/assets/656a79c6-d09e-4ee5-9e3f-0e793be159a3" />

---

## 📈 Training Performance
- **Optimizer**: Adam  
- **Loss Function**: Sparse Categorical Crossentropy  
- **Batch Size**: 32  
- **Epochs**: 50  


---

## ⚙️ Installation & Usage

### 🔧 Prerequisites
- Python 3.8+  
- Jupyter Notebook  
- Required libraries:  
  ```bash
  pip install tensorflow keras numpy pandas matplotlib scikit-learn opencv-python
  ```

### 🚀 Run in Jupyter Notebook
1. Clone the repository:
   ```bash
   git clone https://github.com/YourUsername/Celebrity-Face-Recognition.git
   cd Celebrity-Face-Recognition
   ```

2. Open Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

3. Run the notebook file:
   - `celebrity_face_recognition.ipynb`

4. (Optional) Use the trained model:
   ```python
   from tensorflow.keras.models import load_model
   model = load_model("models/face_model.h5")
   ```

---

## 📜 Training Logs
Training and validation logs are included in the notebook as graphs for:  
- Accuracy vs Epochs  
- Loss vs Epochs  

---

## 🔮 Future Enhancements
- Extend dataset to include more celebrities.  
- Improve accuracy by tuning CNN architecture.  
- Deploy model as a **Flask/Django web app**.  
- Add **real-time webcam face recognition**.  
- Integrate **transfer learning (VGG16, ResNet50)** for higher accuracy.  

---

## 📄 License
This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.  

---
👤 **Author:** 
Linda Lance

