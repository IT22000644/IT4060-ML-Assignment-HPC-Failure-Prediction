# IT4060 – HPC Failure Prediction using Machine Learning

## 📌 Project Overview

This project focuses on **predicting node-level failures in high-performance computing (HPC) systems** using supervised machine learning techniques. The objective is to identify **early warning signals** from system logs that indicate an increased risk of failure.

This work is conducted as part of the **IT4060 Machine Learning assignment**.

---

## 👥 Team Members

| No | Student Name | Student ID | Email       |
| -- | ------------ | ---------- | ----------- |
| 1  | Dikkumbura M N    | IT22000644   | it22000644@my.sliit.lk |
| 2  | Kingsley C. N.   | IT22051448   | it22051448@my.sliit.lk |
| 3  | Rajapaksha W.R.V.A.M.G   | IT22352330   | it22352330@my.sliit.lk |
| 4  | Perera D.J.S   | IT22347794   | it22347794@my.sliit.lk |

---

## 🎯 Problem Statement

Large-scale computing systems such as HPC clusters experience failures due to hardware faults, system overload, and operational issues. These failures can lead to downtime and reduced system efficiency.

This project aims to:

* Predict failures using historical HPC system logs
* Identify patterns preceding failure events
* Develop supervised ML models for early failure detection

---

## 🧠 Research Area

* Computer Systems Reliability
* High-Performance Computing (HPC)
* Predictive Maintenance (PHM)
* Applied Machine Learning

---

## 📊 Dataset

### 🔹 Primary Dataset – LANL HPC Failure Dataset (1996–2005)

* **Source:** https://www.lanl.gov/engage/organizations/aldsct/hpc/usrc/data
* **Dataset Name:** *All systems failure/interrupt data (1996–2005)*

### Description

This dataset contains historical failure and interrupt records from HPC systems at Los Alamos National Laboratory.

The data includes:

* Failure events
* Interrupt events
* System/node identifiers
* Event timestamps

---

## ⚙️ Machine Learning Approach

### Type

* Supervised Learning (Classification)

### Objective

Predict whether a system/node will fail within a future time window (**horizon-based failure prediction**)

---

## 🤖 Algorithms Used

* Logistic Regression (Baseline model)
* Random Forest (Non-linear ensemble model)
* Support Vector Machine (SVM)
* K-Nearest Neighbors (KNN)

---

## 📈 Evaluation Metrics

* Recall
* Precision
* F1-score
* ROC-AUC
* Confusion Matrix

---

## 🔧 Data Preprocessing

* Data cleaning
* Time-based event ordering
* Feature engineering (rolling statistics, trends)
* Label creation (failure within prediction horizon)
* Handling class imbalance
* Time-based train/test split

---

## 🛠️ Tech Stack

* Python
* Jupyter Notebook
* Pandas, NumPy
* Scikit-learn
* Matplotlib / Seaborn

---

## 📝 Detailed Report

* Full workflow report: [report/ML_assignment_report.md](report/ML_assignment_report.md)

---

## 📂 Project Structure

```
IT4060-hpc-failure-prediction/
│
├── data/
├── notebooks/
├── results/
├── report/
├── README.md
└── requirements.txt
```

---

## 📎 References

* LANL Dataset: https://www.lanl.gov/engage/organizations/aldsct/hpc/usrc/data

---
