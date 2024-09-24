# Customer Churn Prediction

This repository contains a project focused on predicting customer churn for a bank's credit card services. The dataset includes a variety of features such as customer age, credit card limit, credit card category, and more. This model is designed to help the bank identify customers at risk of churn so that proactive measures can be taken to retain them.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Technologies Used](#technologies-used)
3. [Dataset](#dataset)
4. [Data Preprocessing](#data-preprocessing)
5. [Modeling](#modeling)
6. [Evaluation](#evaluation)
7. [Results](#results)
8. [Installation and Setup](#installation-and-setup)
9. [Usage](#usage)
10. [Contributing](#contributing)
11. [License](#license)

---

## Project Overview

The purpose of this project is to predict customer churn for a bank. The model uses a variety of supervised learning techniques such as Random Forest, Logistic Regression, and Support Vector Machines (SVM) to identify customers who are likely to discontinue their credit card services. 

The churn dataset contains 10,000 records with around 18 features such as age, salary, marital status, and more. Due to an imbalance in the dataset where only 16.07% of customers churned, we used the SMOTE (Synthetic Minority Oversampling Technique) to balance the data.

## Technologies Used

- Python
- Jupyter Notebooks (Google Colab)
- Plotly for visualizations
- Pandas for data manipulation
- Scikit-learn for machine learning models
- SMOTE for data balancing
- Seaborn and Matplotlib for visualizing data

## Dataset

The dataset used in this project contains customer information such as:
- Customer Age
- Gender
- Dependent count
- Education Level
- Income Category
- Credit Limit
- Card Category
- And other features...

The target variable is the `Attrition_Flag`, which indicates whether a customer has churned or not.

## Data Preprocessing

1. **Feature Selection**: Categorical features such as Gender, Marital Status, Education Level, etc., were one-hot encoded to convert them into numerical representations.
2. **Handling Imbalanced Data**: We used SMOTE to oversample the minority class (`Churned Customers`), as only 16.07% of the customers had churned.
3. **Dimensionality Reduction**: Principal Component Analysis (PCA) was used to reduce the number of dimensions in the dataset, speeding up model training and reducing noise in the data.

## Modeling

The following machine learning algorithms were trained and evaluated:
- Random Forest Classifier
- AdaBoost Classifier
- Support Vector Machines (SVM)
- Logistic Regression
- Gradient Boosting Classifier

Each model was tuned using cross-validation to find the best hyperparameters and was evaluated on test data using the F1 Score.

## Evaluation

The models were evaluated using:
- **F1 Score**: Balancing precision and recall
- **Confusion Matrix**: To understand the true positives, false positives, etc.
- **Precision-Recall Curve**: To visualize the trade-off between precision and recall
- **Cross-Validation**: To ensure the model generalizes well

## Results

- The Random Forest Classifier provided the best F1 Score on the upsampled data.
- The AdaBoost and SVM also performed well, showing competitive results.
- The models struggled slightly on the original unbalanced dataset, but SMOTE helped improve prediction accuracy.

## Installation and Setup

### Prerequisites

You need to have the following installed:
- Python 3.x
- Jupyter Notebook (or Google Colab)
- Required Python libraries (see below)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/customer-churn-prediction.git
   cd customer-churn-prediction

