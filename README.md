# Health-Insurance-Prediction-System
A Supervised Machine Learning Model to Predict Insurance Cost


# 🛡️ InsureIQ · Medical Insurance Premium Predictor

An interactive web application that predicts annual medical insurance charges based on a user's personal and lifestyle profile, powered by a tuned **Random Forest Regressor** (R² ≈ 0.86).

---

## 📸 Features

- **Instant premium estimation** — enter your details and get a predicted annual charge in seconds
- **Risk level meter** — visual Low / Moderate / High / Very High risk indicator
- **Animated UI** — smooth CSS transitions, hero banner, scrolling ticker, and result card animations
- **Fully local** — no external API calls; prediction runs entirely on your machine

---

## 🗂️ Project Structure

```
├── app.py               # Streamlit web application
├── model.pkl            # Trained Random Forest model (joblib)
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

---

## ⚙️ Model Details

| Property | Value |
|---|---|
| Algorithm | Random Forest Regressor |
| Tuning | GridSearchCV (3-fold CV) |
| Best params | `max_depth=10`, `max_features=0.8`, `min_samples_leaf=2`, `min_samples_split=5`, `n_estimators=300` |
| R² (test set) | **0.8599** |
| MSE (test set) | 0.1259 |
| Target transform | `np.log1p(charges)` → inverse with `np.expm1` |

### Input Features

| Feature | Type | Description |
|---|---|---|
| `age` | int | Age of the beneficiary |
| `bmi` | float | Body Mass Index (kg/m²) |
| `children` | int | Number of dependants (0–5) |
| `sex_male` | binary | 1 if male, 0 if female |
| `smoker_yes` | binary | 1 if smoker, 0 otherwise |
| `region_northwest` | binary | US region one-hot |
| `region_southeast` | binary | US region one-hot |
| `region_southwest` | binary | US region one-hot |
| `bmi_age_interaction` | float | `bmi × age` engineered feature |
| `bmi_category_Obese` | binary | BMI ≥ 30 |
| `bmi_category_Overweight` | binary | 25 ≤ BMI < 30 |
| `bmi_category_Underweight` | binary | BMI < 18.5 |

> **Note:** `region_northeast` is the reference (dropped) category. `bmi_category_Normal` is the reference BMI category.

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/insureiq.git
cd insureiq
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add the model file

Place your `model.pkl` file in the root of the project directory (same folder as `app.py`).

### 4. Run the app

```bash
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`.

---

## ☁️ Deploy to Streamlit Community Cloud

1. Push the repo (including `model.pkl`) to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your repo
3. Set **Main file path** to `app.py`
4. Click **Deploy** — Streamlit will install `requirements.txt` automatically

> ⚠️ `model.pkl` must be committed to the repo for cloud deployment to work. If the file is large, consider Git LFS.

---

## 📦 Dependencies

| Package | Version |
|---|---|
| streamlit | ≥ 1.35.0 |
| scikit-learn | ≥ 1.4.0 |
| pandas | ≥ 2.1.0 |
| numpy | ≥ 1.26.0 |
| joblib | ≥ 1.3.0 |

---

## ⚠️ Disclaimer

InsureIQ provides **estimates only** based on a machine learning model trained on a public dataset. Predictions are not financial, medical, or actuarial advice and should not be used as the basis for any insurance decisions.

---

## 📄 License

MIT License — free to use, modify, and distribute.
