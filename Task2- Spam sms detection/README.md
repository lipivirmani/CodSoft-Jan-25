# 📩 SMS Spam Detection using Machine Learning  

## 🚀 Project Overview  
This project builds an **AI-based SMS Spam Classifier** that detects whether a message is **Spam** or **Ham (Legitimate)** using **TF-IDF, Naive Bayes, Logistic Regression, and Support Vector Machines (SVM)**.  

🔹 **Dataset**: [UCI ML SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)  
🔹 **Techniques Used**: Text Preprocessing, TF-IDF, Machine Learning  
🔹 **ML Models**: Naive Bayes, Logistic Regression, Random Forest, SVM  
🔹 **Key Features**: Handles class imbalance, improves spam detection  
🔹 **Visualizations**: Pie chart, confusion matrix ,Model Comparison Bar Graph

---

## 📂 Dataset Information  
The dataset consists of **5,572 messages**, labeled as:  
- **ham** (legitimate) – 4,827 messages  
- **spam** (fraudulent) – 745 messages  

---

## 🛠 Technologies Used  
- **Python** (Pandas, NumPy, Scikit-learn)  
- **NLTK** (Text Processing, Stopwords)  
- **Matplotlib & Seaborn** (Data Visualization)  

---

## 📊 Data Visualization  
1️⃣ **Distribution of Messages**  
📌 **Pie Chart** showing the proportion of spam and ham messages.  


2️⃣ **Confusion Matrix**  
📌 **Evaluates Model Performance** using accuracy, precision, recall, and F1-score.  

3️⃣ **Bar Graph** for  Model-wise comparison (Precision, Recall, F1-score, Accuracy)

---

## 📌 Installation  
Clone the repository and install dependencies:  
```bash
git clone https://github.com/lipivirmani/spam-classifier.git
cd spam-classifier
pip install -r requirements.txt
```

---

## 📝 Usage  
### 1️⃣ **Train the Model**
```python
python train.py
```
### 2️⃣ **Predict a New SMS**
```python
message = "Congratulations! You won a free lottery ticket."
print(predict_message(message))
```
Expected Output:
```
{'Naive Bayes': 1, 'Logistic Regression': 1, 'SVM': 1, 'Random Forest': 1}
```

---

## 📈 Model Performance  
| Model  | Accuracy | Precision | Recall | F1 Score |
|--------|----------|-----------|--------|----------|
| Naive Bayes | 97.8% | 91.2% | 88.6% | 89.9% |
| Logistic Regression | 98.1% | 92.5% | 90.3% | 91.4% |
| SVM | 98.6% | 94.2% | 92.1% | 93.1% |
| Random Forest | 98.4% | 93.8% | 91.7% | 92.7% |

✅ **SVM gives the best performance!** 🚀

---

### **🚀 Future Enhancements**  
🔹 Implement **Deep Learning models** (LSTMs, Transformers)  
🔹 Add **real-time SMS filtering API**  
🔹 Deploy as a **web application**  

---

## 📜 License  
This project is licensed under the **MIT License**.  
```
MIT License
Copyright (c) 2025 [Lipi Virmani]
Permission is granted to use, copy, modify, and distribute this software.
Project developed as part of Machine Learning internship at CODSOFT
```

---
