# W207: Introduction to Machine Learning

## Course Overview
Introduces students to the core concepts, algorithms, and practical workflows used in modern machine learning systems. Focuses on supervised and unsupervised learning techniques, model evaluation, feature engineering, and E2E process of building predictive models from real-world data.

## Learning Objectives
- Formulate machine learning problems: identifying supervised/unsupervised problems, regression vs. classification, appropriate evaluation metrics
- Classification models: logistic regression, decision trees, random forests, gradient boosting, support vector machines
- Practice train/test splitting, cross-validation, hyperparameter tuning, precision/recall tradeoffs
- Apply feature engineering techniques: encoding categorical variables, feature scaling, dimensionality reduction, missing value strategies

## Folder Structure

```
W207-Intro-to-Machine-Learning/
├── code/       # Scripts, notebooks, and source code
├── data/       # Raw and processed datasets
└── reports/    # Written reports, papers, and deliverables
```

## Final Project
Choose a public dataset of interest, identify a problem that machine learning could resolve, then apply the proper techniques.

Our project sought to use a electronic health records (EHRs) to predict patients' likelihood of developing diabetes. Medical and demographic fields included gender, age, smoking history, HbA1c levels, among other indicators. Applied logistics regression, k-nearest neighbors, and decision tree classifiers along with random forest, adaboost, and gradient boost as ensemble methods.

## Notes
Key libraries used: sklearn (Logistic Regression, KNeighbors Classifier, DecisionTreeClassifier) , Pandas, NumPy, seaborn, matplotlib
