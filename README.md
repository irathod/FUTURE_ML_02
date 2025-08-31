🏦 Bank Customer Churn Analytics Dashboard
This project, completed as part of my Machine Learning Internship with Future Interns, focuses on predicting customer churn for a banking dataset. The goal was to develop a predictive system and an analytics dashboard that provides actionable insights to business decision-makers.

🚀 Project Overview
Explored and Pre-processed the Bank Customer Churn dataset from Kaggle.

Engineered features such as tenure, balance, credit score, age, and product usage to prepare the data for modeling.

Built and compared churn prediction models using Logistic Regression, Random Forest, and XGBoost.

Evaluated model performance using key metrics like Accuracy, Precision, Recall, F1-Score, and ROC-AUC.

Designed an interactive dashboard in Power BI to present key churn insights and visualize the results.

📂 Features
✅ Predict Churn Probability: The model can predict the likelihood of each customer churning.
✅ Actual vs Predicted Churn: Provides a clear view of the model's classification performance.
✅ KPI Cards: Displays key metrics like Total Customers, Churned Customers, Churn Rate, and Model Accuracy.
✅ Feature Importance: Visualizes which factors (e.g., Age, Balance) have the most impact on churn.
✅ Segmentation Analysis: Allows for in-depth analysis of churn rates by Gender, Geography, Tenure, and Balance.
✅ Model Evaluation: Includes a Confusion Matrix and ROC-AUC chart for a comprehensive performance review.

🛠️ Tech Stack
Python: Used for the entire machine learning pipeline, from data cleaning to model building.

Pandas, NumPy: For data manipulation and numerical operations.

Scikit-learn, XGBoost: For building, training, and evaluating the classification models.

Matplotlib, Seaborn: For data visualization within the Python environment.

Power BI: Used to create the final interactive dashboard for business intelligence.

📈 Model Performance
Based on typical model runs for this dataset, here are some sample performance metrics. Remember to replace these with your actual results after running your models.

Logistic Regression: ~81% Accuracy

Random Forest: ~86% Accuracy

XGBoost: ~87% Accuracy (best performing)

ROC-AUC Score: ~85%

<img width="1158" height="656" alt="bank customer analysis dashboard png" src="https://github.com/user-attachments/assets/9035f7cc-4d9c-4d5e-bd60-b64b7a767e8a" />

🔗 Repository Contents
bank_churn_clean.csv: The cleaned and pre-processed dataset used for modeling.

churn_model.ipynb: The Jupyter notebook containing the entire machine learning pipeline.

task2.pbix: The Power BI Dashboard file.

🏆 Internship Task
This project is part of Task 2 under my internship at Future Interns, focusing on Churn Prediction & Customer Analytics.
