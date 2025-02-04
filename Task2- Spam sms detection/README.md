
# **📩 SMS Spam Detection using Machine Learning**  

## **Overview**  
This project is an **SMS Spam Detection System** that classifies messages as **spam** or **ham** (not spam) using **Natural Language Processing (NLP)** and **Machine Learning algorithms**.  

The dataset used is the **[UCI SMS Spam Collection](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)**.  

## **🔍 Features**  
✔️ **Text preprocessing**: Removing punctuations, stopwords, and special characters.  
✔️ **TF-IDF Vectorization**: Converting text into numerical form.  
✔️ **Machine Learning Models Used**:  
   - Logistic Regression  
   - Random Forest  
   - Gradient Boosting  
✔️ **Prediction on New Messages**: Classifies new SMS as spam or ham.  

---

## **📂 Dataset Information**  
The dataset contains **5,574 SMS messages**, each labeled as either **spam (1)** or **ham (0)**.  

| Feature | Description |  
|---------|------------|  
| `text` | The SMS message |  
| `label` | "spam" or "ham" (1 or 0) |  

---

## **📊 Visualizations**  
This project includes various data visualizations for better insights:  

### **1️⃣ Spam vs. Ham Distribution (Pie Chart)**  
📌 Chart showing the proportion of spam and ham messages.  


### **2️⃣ Frequent Spam-Specific Words (Bar Graph)**  
📌 Displays the most common words in spam messages after removing stopwords.  


### **3️⃣ Word Distribution with KDE Plot**  
📌 Shows the density distribution of word count for spam and ham messages.  


---

## **🛠 Model Training & Evaluation**  
The dataset was split into **training (80%)** and **testing (20%)** sets. Three models were trained:  
✔️ **Logistic Regression**  
✔️ **Random Forest**  
✔️ **Gradient Boosting**  

Performance was evaluated using **Accuracy, Precision, Recall, and F1-Score**.

Example:  
```python
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
```

---

## **🔮 Predicting a New Message**  
The trained model can predict whether a new message is **spam or ham**.  

```python
message = ["Hey, How are you"]
message_transformed = vectorizer.transform(message)
prediction = model.predict(message_transformed)
print("Spam" if prediction == 1 else "Ham")
```
✅ **Output:** `"Ham"`  

---

## **📌 How to Run the Project**  
1️⃣ **Clone the repository**  
```bash
git clone https://github.com/lipivirmani/SMS-Spam-Detection.git
cd SMS-Spam-Detection
```
2️⃣ **Install dependencies**  
```bash
pip install -r requirements.txt
```
3️⃣ **Run the Jupyter Notebook or Python script**  
```bash
jupyter notebook
```

---

## **📜 Conclusion**  
This project successfully detects **spam messages** using **Machine Learning & NLP**. With additional **hyperparameter tuning and deep learning**, further improvements can be made.  

---

### **🚀 Future Enhancements**  
🔹 Implement **Deep Learning models** (LSTMs, Transformers)  
🔹 Add **real-time SMS filtering API**  
🔹 Deploy as a **web application**  

---
## ** 👩‍💻 Author

Copyright (c) [2025] [Lipi Virmani]

Project developed as part of Machine Learning internship at CODSOFT
