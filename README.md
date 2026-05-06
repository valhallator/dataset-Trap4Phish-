# Phishing Detection in PDF Files - CIC-Trap4Phish 2025

This repository contains a comprehensive cybersecurity data analysis project developed for the Master's in Cybersecurity at **IPVC (Instituto Politécnico de Viana do Castelo)**. The study focuses on detecting phishing attempts hidden in PDF structures using advanced Machine Learning techniques.

## 📊 Project Overview
Using the **CIC-Trap4Phish 2025** dataset from the Canadian Institute for Cybersecurity, this research evaluates how structural features and metadata can distinguish legitimate documents from malicious ones.

### Key Results
*   **Highest Accuracy:** 99.5% (Random Forest)
*   **Best Generalization:** Random Forest (AUC 0.999)
*   **Key Discriminators:** Entropy of streams, Metadata size, and Object count.

## 🛠️ Methodology
The project is organized into four distinct phases:

1.  **Exploratory Data Analysis (f1):** In-depth study of feature distributions, correlation matrices, and unsupervised clustering (K-Means) to identify natural patterns.
2.  **Data Pre-processing (f2):** Implementation of Log1p transformations to handle skewness, outlier clipping, and **SMOTE** (Synthetic Minority Over-sampling Technique) to address class imbalance.
3.  **Machine Learning Modeling (f3):** Comparative analysis of 7 algorithms (Random Forest, KNN, SVM, Decision Tree, Naive Bayes, Logistic and Linear Regressions).
4.  **Simulation & Validation (f4):** Practical testing of models against ambiguous cases to verify robustness.

## 📁 Repository Structure
*   `code_output/`: Modular Python scripts and visual results for each phase.
*   `data/`: Statistical summaries (Original dataset should be obtained from CIC).
*   `MCiber-2025-2026-ADC-CIC-Trap4Phish_2025-iurisousa_v3.pptx`: Final defense presentation.

## 🚀 How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Run phases sequentially: `python code_output/f1/fase1_analise_exploratoria.py`, etc.
