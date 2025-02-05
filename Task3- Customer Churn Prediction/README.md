# Customer Churn Prediction

This project aims to predict customer churn for a subscription-based business using historical customer data. The dataset includes features like usage behavior and customer demographics. The project applies machine learning techniques such as Logistic Regression, Random Forest, and Gradient Boosting to identify potential churners.

## Features and Techniques Used

- **Data Preprocessing:**
  - Handling missing values by replacing numerical features with their mean and categorical features with their mode.
  - Encoding categorical variables using Label Encoding.
  - Standardizing numerical features using StandardScaler.
  - Removing duplicate records.

- **Machine Learning Models:**
  - **Logistic Regression** – A simple yet effective linear model for classification.
  - **Random Forest** – An ensemble learning method that improves accuracy by combining multiple decision trees.
  - **Gradient Boosting** – A boosting algorithm that builds models sequentially to correct previous errors.

## 🛠️ Tech Stack  
- **Python**  
- **Pandas, NumPy**  
- **Scikit-Learn**  
- **Matplotlib, Seaborn**  

- **Model Evaluation Metrics:**
  - Accuracy
  - Precision
  - Recall
  - F1-score

## Visualizations

1. **Model Performance Comparison:** A bar chart comparing the performance of different models on accuracy, precision, recall, and F1-score.
2. **Feature Importance:** A horizontal bar chart displaying the importance of features based on the Random Forest model.
3. **Confusion Matrix:** A heatmap for Gradient Boosting predictions to analyze classification performance.

## Predictions

- The trained models predict whether a new customer is likely to churn based on their demographics and usage behavior.
- Example:
  ```
  Churn Prediction: Not Churn
  ```

## Installation & Usage

1. Clone the repository:
   ```sh
   git clone https://github.com/lipivirmani/customer-churn-prediction.git
   cd customer-churn-prediction
   ```
2. Install required dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Run the Python script:
   ```sh
   python customer_churn_prediction.py
   ```

## License

This project is licensed under the [MIT License](LICENSE).

✨ Author
Developed as part of an internship task at Codsoft
