# 🚀 Optimizing XGBoost for High-Dimensional Text Data paper

## 📌 Project Overview  
This research focuses on optimizing **XGBoost** for **high-dimensional text classification** using the **20 Newsgroups Dataset**. By implementing **feature selection, dimensionality reduction, and hyperparameter tuning**, we aim to enhance model efficiency, reduce overfitting, and improve generalization.  

## 🔑 Key Objectives  
- **Feature Engineering**: Convert text data into numerical form using **TF-IDF** and **Word Embeddings (Word2Vec, FastText)**.  
- **Feature Selection**: Apply **L1-based selection** to remove irrelevant features.  
- **Dimensionality Reduction**: Use **PCA & Latent Semantic Analysis (LSA)** to lower computational overhead.  
- **Hyperparameter Tuning**: Optimize XGBoost parameters like **max_depth, learning_rate, n_estimators** for better accuracy.  
- **Performance Evaluation**: Compare accuracy, training time, and model explainability before and after optimization.  
- **Explainability & Insights**: Use **SHAP values** to interpret model decisions.  

## 📂 Dataset  
- **20 Newsgroups Dataset**: A text dataset containing 20 categories of news articles.  
- Available in **scikit-learn** and can be loaded using:  
  ```python
  from sklearn.datasets import fetch_20newsgroups
  data = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
