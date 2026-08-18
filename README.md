# AI-Powered Detection of Phishing Websites Using URL and Webpage Features

## Project Title

AI-Powered Detection of Phishing Websites Using URL and Webpage Features

## Team Members

| S.No | Roll Number | Student Name |
|------|------------|--------------|
| 1 | 2420030182 | G. Manaswi |
| 2 | 2420030100 | M. Aruna Grace |

## Abstract

Phishing websites are one of the major cybersecurity threats because they imitate legitimate websites to deceive users into revealing sensitive information such as usernames, passwords, banking credentials, and personal details.

This project proposes an AI-powered phishing website detection system using machine learning and website-based features. The system analyzes URL, HTML, hyperlink, and webpage text features to classify websites as either **Phishing** or **Legitimate**.

The project uses multiple datasets, including the D1 phishing webpage dataset, the UCI Phishing Websites Dataset, and the PhiUSIIL Phishing URL Dataset. The data is cleaned and preprocessed before extracting meaningful features.

Machine learning models such as **XGBoost, CatBoost, and LightGBM** will be trained and compared. Feature selection and hyperparameter optimization will be used to improve performance. Explainable AI techniques such as **SHAP or LIME** will also be used to explain why a website is predicted as phishing or legitimate.

The final system will provide a web-based interface where users can submit a URL and receive a phishing detection result along with an explanation.

## Problem Statement

Traditional phishing detection methods such as blacklists and rule-based systems may fail to detect newly created and previously unseen phishing websites.

Therefore, there is a need for an intelligent system that can analyze different website characteristics and automatically identify phishing websites using machine learning.

## Objectives

- To develop an AI-powered system for detecting phishing and legitimate websites.
- To analyze URL, HTML, hyperlink, and webpage text features.
- To use multiple phishing datasets for training and evaluation.
- To enhance detection using additional engineered features.
- To train and compare XGBoost, CatBoost, and LightGBM models.
- To improve model performance using feature selection and hyperparameter optimization.
- To provide explainable predictions using SHAP or LIME.
- To develop a web application for phishing website prediction.

## Datasets

### 1. D1 Phishing Webpage Dataset

Research source:

https://www.nature.com/articles/s41598-022-10841-5

### 2. UCI Phishing Websites Dataset

Dataset link:

https://archive.ics.uci.edu/dataset/327/phishing+websites

### 3. PhiUSIIL Phishing URL Dataset

Dataset link:

https://archive.ics.uci.edu/dataset/967/phiusil+phishing+url+dataset

## Proposed Features

The system will analyze features such as:

- URL length
- Number of special characters
- Number of subdomains
- Suspicious keywords
- URL entropy
- HTML forms
- Scripts and iframes
- Internal and external links
- External-link ratio
- Form-domain mismatch
- Brand similarity
- Webpage text features

## Machine Learning Models

The proposed models include:

- XGBoost
- CatBoost
- LightGBM
- Ensemble learning models

## Explainable AI

The project will use:

- SHAP
- LIME

These techniques help explain which features influenced the prediction.

## Technology Stack

- **Programming Language:** Python
- **Machine Learning:** Scikit-learn
- **Models:** XGBoost, CatBoost, LightGBM
- **Explainable AI:** SHAP / LIME
- **Backend:** Flask or FastAPI
- **Frontend:** HTML, CSS, JavaScript
- **Database:** SQLite or MySQL
- **Version Control:** Git and GitHub

## Project Workflow

Dataset Collection  
↓  
Data Cleaning and Preprocessing  
↓  
Feature Extraction  
↓  
Enhanced Feature Engineering  
↓  
Feature Selection  
↓  
Model Training  
↓  
Hyperparameter Optimization  
↓  
SHAP / LIME Explainability  
↓  
Phishing / Legitimate Prediction  
↓  
Web Application and Database  
↓  
Performance Evaluation and Comparison

## Folder Structure

```text
KLH-CSE-2026-27-2420030182-Phishing-URL-s/
│
├── src/        # Source code
├── docs/       # Abstract and project documents
├── data/       # Dataset files or dataset references
├── results/    # Model results and outputs
├── reports/    # Review documents and reports
└── README.md    # Project information
