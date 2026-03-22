# 🎓 Student At-Risk Early Warning System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-Machine%20Learning-F7931E)
![Render](https://img.shields.io/badge/Deployed-Render-46E3B7)

**Live Demo:** [Student Risk Predictor Dashboard](https://student-risk-predictor-2.onrender.com/)

A Machine Learning-based web application designed for the **Indian School Context** (CBSE/ICSE/State Board). This system predicts the likelihood of a student failing or being "at-risk" based on their academic performance, attendance, socio-economic background, and study habits.



---

## ✨ Features

- **🎯 Predictive Modeling:** Uses a Random Forest Classifier trained on synthetic data mapping to realistic Indian student demographics.
- **📊 Real-time Risk Score:** Calculates a personalized probability (%) of a student failing or dropping out.
- **🔍 SHAP Explainability:** Transparent predictions showing exactly *why* a student is at risk (factors increasing risk vs. protective factors), using `shap` TreeExplainer.
- **📚 Subject-Specific Context:** Dynamically adjusts to the Indian curriculum based on Class (9-12) and Stream (Science PCM/PCB, Commerce, Arts/Humanities).
- **📝 Automated Teacher Recommendations:** Generates actionable, context-aware suggestions for educators (e.g., remedial classes, scholarship suggestions, counseling).
- **🔗 Educational Resources:** Direct links to free Indian Government resources (DIKSHA, NCERT, PM e-Vidya).

---

## 🛠️ Technology Stack

- **Frontend & UI:** [Streamlit](https://streamlit.io/)
- **Machine Learning Profile:** [Scikit-learn](https://scikit-learn.org/) (Random Forest Model)
- **Model Explainability:** [SHAP](https://shap.readthedocs.io/) (SHapley Additive exPlanations)
- **Data Manipulation:** Pandas, NumPy
- **Deployment:** Render (`render.yaml` & `Procfile` configured)

---

## 🚀 Running the App Locally

### 1. Clone the repository
```bash
git clone https://github.com/Legends0024/student-risk-predictor.git
cd student-risk-predictor
```

### 2. Create a virtual environment (Optional but Recommended)
```bash
python -m venv .venv
# On Windows
.venv\Scripts\activate
# On Mac/Linux
source .venv/bin/activate
```

### 3. Install Requirements
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Dashboard
```bash
streamlit run app/dashboard_indian.py
```

---

## 📂 Project Structure

```text
student-risk-predictor/
├── app/
│   └── dashboard_indian.py      # Main Streamlit application
├── data/                        # Datasets (Training and Evaluation data)
├── notebooks/                   # Jupyter notebooks for EDA and Model Training
├── outputs/
│   └── models/
│       └── random_forest.pkl    # Serialized trained Random Forest model
├── requirements.txt             # Python dependencies
├── Procfile                     # Render deployment configuration
└── render.yaml                  # Infrastructure as Code for Render
```

---

## 🧠 How the Model Works

1. **Input Data**: The application collects details regarding School Type, Board, Attendance, Study Hours, Family Income, Internet Access, and raw Subject Marks.
2. **Preprocessing**: Categorical data is mapped and encoded automatically under the hood to align with the training data requirements.
3. **Inference**: The `RandomForestClassifier` outputs a trajectory probability. An overarching rule applies: **Any subject score < 33 OR overall percentage < 33% = immediate At-Risk classification**.
4. **SHAP Analysis**: SHAP isolates the top contributing features to explain the model's decision-making process.


*Feel free to star ⭐ this repository if you find it helpful!*
