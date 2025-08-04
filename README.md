
# Anemia Detection System — *AnemiaSense*

**AnemiaSense** is a Python-based machine learning application that helps users **predict anemia status** using clinical input data. With a simple Flask-powered web interface and real-time prediction results, this app offers a quick and effective way to assist in the early recognition and management of anemia.

---

## Features

* Predict anemia using **health parameters** like hemoglobin, RBC count, and more
* Enter patient data via a **clean, user-friendly form**
* Get **instant prediction** (Anemic / Not Anemic)
* Highlights the **importance of each feature** through trained ML models
* Responsive web UI styled with HTML/CSS
* Automatically resets form and updates results smoothly

---

## Tech Stack

* **Python 3**
* **Flask** for web deployment
* **Scikit-learn** for ML model training
* **Pandas** and **NumPy** for data preprocessing
* **Matplotlib** and **Seaborn** for visualizations
* **Pickle** for model serialization

---

## Dataset (anemia\_dataset.csv)

The dataset used contains the following medical features:

* `Age`
* `Hemoglobin`
* `MCV` (Mean Corpuscular Volume)
* `MCH` (Mean Corpuscular Hemoglobin)
* `MCHC` (Mean Corpuscular Hemoglobin Concentration)
* `RBC Count`

---

## How It Works

1. User enters patient health parameters into the web form
2. Input is passed to a trained **Decision Tree Classifier**
3. The model processes and classifies the input as **Anemic** or **Not Anemic**
4. The result is instantly displayed on the same page
5. Background opacity is adjusted and the form resets automatically

---


Let me know if you want this in `.md` file format or want to include screenshots, GitHub badge, or deployment instructions (like Render, Railway, etc.).
