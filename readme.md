🚀 Churn Prediction & Salary Regression using ANN








📌 Project Overview

This project focuses on predicting customer churn and estimating customer salary using Artificial Neural Networks (ANN).
Two separate deep learning models are developed to serve different business objectives:

Churn Classification Model → Identifies customers who are likely to leave.

Salary Regression Model → Predicts customer salary based on input features.

The complete solution is deployed using Streamlit, providing an interactive and user-friendly web interface.

🎯 Objective:
To deliver actionable insights for customer retention strategies and salary prediction using deep learning models.

✨ Key Features
🔍 Churn Prediction (Classification)

Accepts customer demographic and behavioral inputs

Outputs:

Churn probability

Binary decision: Churn / Not Churn

Helps businesses identify high-risk customers

💰 Salary Prediction (Regression)

Uses customer features to predict salary

Outputs a continuous numerical salary value

Useful for analytical and decision-making purposes

🖥️ Interactive Web Interface

Built with Streamlit

Simple, clean, and intuitive UI

Users can:

Enter customer details

Get instant churn and salary predictions

📊 Dataset Description

Customer-centric dataset containing:

Demographic features

Behavioral metrics

Salary information

Churn label

Target Variables

Churn → Classification target

Salary → Regression target

Data Preprocessing Steps

Handling missing values

Encoding categorical variables

Feature scaling (crucial for ANN performance)

Train–test split for evaluation

🧠 Model Architecture & Details
1️⃣ Churn Classifier (ANN)

Model Type: Artificial Neural Network (Classifier)

Hidden Layers: Fully connected (Dense)

Activation Functions:

Hidden Layers → ReLU

Output Layer → Sigmoid

Loss Function: Binary Crossentropy

Optimizer: Adam

Evaluation Metric: Accuracy

📈 Used to classify customers into churn / non-churn categories.

2️⃣ Salary Regressor (ANN)

Model Type: Artificial Neural Network (Regressor)

Hidden Layers: Fully connected (Dense)

Activation Functions:

Hidden Layers → ReLU

Output Layer → Linear

Loss Function: Mean Squared Error (MSE)

Optimizer: Adam

Evaluation Metrics:

RMSE (Root Mean Squared Error)

MAE (Mean Absolute Error)

R² Score

📉 Designed to predict continuous salary values.

🛠️ Installation & Setup
Clone the Repository
git clone <repository-url>
cd churn_salary_ANN

Install Dependencies
pip install -r requirements.txt

▶️ Running the Application

Launch the Streamlit app using:

streamlit run app.py


Then open your browser and navigate to:

http://localhost:8501


✔️ Enter customer data
✔️ Get churn prediction
✔️ Get salary estimation

📂 Project Structure
churn_salary_ANN/
│
├── data/                   # Dataset files
├── models/                 # Trained ANN models (classifier & regressor)
├── app.py                  # Streamlit application
├── utils.py                # Helper & preprocessing functions
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation

📦 Dependencies

Python 3.x

TensorFlow / Keras

Streamlit

Pandas

NumPy

Scikit-learn

🚀 Future Enhancements

Hyperparameter tuning using:

GridSearchCV

Keras Tuner

Add visual evaluation metrics in Streamlit (loss curves, R² plots)

Integrate explainability tools:

SHAP

LIME

Improve regression performance using ensemble or hybrid models

Cloud deployment (Streamlit Cloud / AWS / Azure)

👨‍💻 Author

Karamjodh Singh
Machine Learning Engineer | AI & ML Enthusiast
B.Tech CSE (AI & ML) – Chandigarh University

⭐ Final Note

This project demonstrates the end-to-end ML workflow:

Data preprocessing

ANN modeling (classification & regression)

Model evaluation

Web deployment using Streamlit

If you find this project useful, feel free to ⭐ star the repository!