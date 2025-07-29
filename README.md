
# ❤️ Heart Disease Prediction

This project aims to predict the presence of heart disease using a medical dataset obtained from [Kaggle](https://www.kaggle.com). The dataset contains various features such as age, cholesterol, blood pressure, heart rate, and more. The goal is to help identify whether a person is likely to have heart disease.

---

## 📊 Models Used

I trained the dataset using two machine learning classification models:

### 🔹 1. Logistic Regression

- **Accuracy:** 83.05%
- **Classification Report:**

```
               precision    recall  f1-score   support

           0       0.90      0.79      0.84        34
           1       0.76      0.88      0.81        25

    accuracy                           0.83        59
   macro avg       0.83      0.84      0.83        59
weighted avg       0.84      0.83      0.83        59
```

- **Confusion Matrix:**

```
[[27  7]
 [ 3 22]]
```

---

### 🔹 2. Ridge Classifier ✅ (Selected)

- **Accuracy:** 91.52%
- **Classification Report:**

```
               precision    recall  f1-score   support

           0       0.91      0.94      0.93        34
           1       0.92      0.88      0.90        25

    accuracy                           0.92        59
   macro avg       0.92      0.91      0.91        59
weighted avg       0.92      0.92      0.92        59
```

- **Confusion Matrix:**

```
[[32  2]
 [ 3 22]]
```

✅ Since RidgeClassifier performed better, it was selected for deployment.

---

## 🧠 Streamlit App

Experience the live deployment of this machine learning-powered web app that predicts the risk of heart disease based on user-provided health inputs. The application, built using Streamlit, leverages a clean and interactive interface for easy data input, real-time prediction, and basic visualizations.

🔗 Try it here: Heart Disease Prediction App
---

## 📁 Dataset

- Dataset used: **Heart Disease UCI Cleveland**
- Source: [Kaggle - Heart Disease Dataset](https://www.kaggle.com/datasets)

---



---

## 📬 Contact

For questions or suggestions, feel free to reach out.

---
