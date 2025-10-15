🩺 Breast Cancer Diagnosis using Machine Learning
📘 Overview

This project applies supervised machine learning algorithms to predict whether a breast tumor is malignant or benign based on diagnostic features.
It compares multiple classification models and evaluates their performance using metrics such as accuracy, precision, recall, F1-score, and ROC-AUC.

📂 Dataset

Source: Kaggle – Breast Cancer Wisconsin (Diagnostic) Data Set

Format: CSV file with numerical features

Target column: diagnosis (M = malignant, B = benign)

Total samples: 569

Features: 30 numeric input features derived from cell nuclei measurements

⚙️ Algorithms Used
Algorithm	Description
Logistic Regression	Linear classifier for binary diagnosis prediction
Naive Bayes	Probabilistic model based on Bayes’ theorem
Random Forest Classifier	Ensemble of decision trees for robust classification
XGBoost Classifier	Optimized gradient boosting model for high accuracy
🧹 Data Preprocessing

Loaded the Kaggle dataset and inspected for missing values

Encoded categorical target labels (M, B) into binary format

Performed feature scaling using StandardScaler

Split dataset into training (80%) and testing (20%) sets

🧠 Model Training & Evaluation

Each classifier was trained on the training set and evaluated on the test set.
Metrics computed:

Accuracy

Precision

Recall

F1-score

ROC–AUC Score

The ROC–AUC curve was plotted to visualize model performance and discriminative power.

🧾 Results Summary
Model	Accuracy	ROC-AUC
Logistic Regression	~98%	0.99
Naive Bayes	~96%	0.98
Random Forest	~98%	0.99
XGBoost	~99%	1.00

(Values may slightly vary depending on random seed and preprocessing.)

💻 Installation & Execution

1️⃣ Clone the repository
git clone https://github.com/<ankita848>/Breast_Cancer_Detction.git

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Run the notebook

Open Jupyter Notebook or VS Code:

jupyter notebook Breast_Cancer_Diagnosis.ipynb

📦 Requirements
pandas
numpy
scikit-learn
matplotlib
xgboost

📚 Key Learnings

Feature scaling importance in model comparison

Ensemble and boosting methods improve predictive accuracy

ROC–AUC is a robust metric for binary medical diagnosis problems

 Author
 
Ankita Das

 
