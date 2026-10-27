# Power_Consumption_Prediction
# ⚡ Power Consumption Prediction using Machine Learning

## 📌 Project Overview

This project uses Machine Learning to predict power consumption based on input features from a power consumption dataset. It uses a trained **Random Forest Regressor** to generate predictions and provides an interactive interface built with Streamlit.

The goal is to demonstrate how machine learning can be used to analyze energy consumption patterns and estimate power usage from relevant input data.

## 🎯 Objectives

- Predict power consumption using a trained machine learning model.
- Explore and preprocess the dataset for model training.
- Apply Random Forest Regression for prediction.
- Build an interactive web application using Streamlit.
- Make machine learning predictions accessible through a simple user interface.

## 🛠️ Technologies Used

- **Python**
- **Pandas** — data manipulation and analysis
- **NumPy** — numerical operations
- **Scikit-learn** — machine learning and preprocessing
- **Matplotlib / Seaborn** — data visualization, if used
- **Streamlit** — web application
- **Git & GitHub LFS** — version control and large model storage

## 🤖 Machine Learning Model

**Algorithm:** Random Forest Regression

Random Forest Regression combines predictions from multiple decision trees to produce a final numerical prediction. It can model nonlinear relationships between input features and power consumption.

The project uses a previously trained model and its associated preprocessing artifacts to generate predictions.

## 📂 Project Structure

```text
Power_Consumption_Prediction/
│
├── app.py
├── random_forest_model.pkl
├── scaler.pkl
├── requirements.txt
├── .gitattributes
└── README.md
```

*Note: Update this structure if your repository contains additional files or uses different filenames.*

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Rajak20/Power_Consumption_Prediction.git
```

### 2. Navigate to the project directory

```bash
cd Power_Consumption_Prediction
```

### 3. Create a virtual environment (recommended)

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
.\venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

Make sure Git LFS is installed and configured if the model files are tracked using Git LFS.

### 5. Run the Streamlit application

```bash
streamlit run app.py
```

The application will open in your browser, usually at `http://localhost:8501`.

## 🚀 Live Demo

🔗 **Streamlit App:** [Add your deployed application URL here]

## 📈 Key Features

- Interactive power consumption prediction.
- Pre-trained Random Forest model integration.
- Input preprocessing using a saved scaler, where required.
- User-friendly Streamlit interface.
- Reusable model artifacts for making predictions.

## 📚 Learning Outcomes

Through this project, I gained practical experience in:

- Building and using regression models.
- Loading saved machine learning models.
- Applying preprocessing consistently during inference.
- Integrating machine learning with Streamlit.
- Managing large model files with Git LFS.
- Deploying machine learning applications.

## 🔮 Future Improvements

- Compare Random Forest with Linear Regression, Decision Tree, and Gradient Boosting.
- Add visualizations of historical power consumption.
- Evaluate the model using MAE, RMSE, and R².
- Add time-based forecasting if timestamped data is available.
- Improve the interface with prediction history and downloadable results.

## 👨‍💻 Author

**Rajak M**

B.Tech in Information Technology | Aspiring AI/ML Engineer

🔗 **GitHub:** [Rajak20](https://github.com/Rajak20)

---

⭐ If you find this project useful, consider giving the repository a star!
