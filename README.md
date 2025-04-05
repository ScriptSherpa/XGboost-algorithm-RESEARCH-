
# 📈 Stock Market Direction Prediction using Technical Indicators and XGBoost

Predicting the short-term direction of stock prices is challenging, but with the help of technical indicators and machine learning models like XGBoost, we attempt to forecast whether the **next day's price will go up or down**.

---

## 🔍 Project Overview

This project focuses on **binary classification** of stock price direction using historical data of **Apple Inc. (AAPL)**. It includes:

- Technical Indicator Engineering
- Data Visualization
- Feature Selection
- Classification using **XGBoost**
- Performance Tuning & Evaluation

---

## 📁 Dataset

We used historical OHLCV (Open, High, Low, Close, Volume) data for AAPL.

- **Original Shape:** `(1410, 8)`
- Features like `Daily Return`, `SMA`, `EMA`, `MACD`, `RSI`, etc., were derived.
- Final dataset shape after feature engineering: `(1311, 19)`
- After volatility filtering: `(966, 20)`

---

## 🧠 Features Engineered

| Feature | Description |
|--------|-------------|
| SMA_20 | 20-day Simple Moving Average |
| SMA_100 | 100-day Simple Moving Average |
| EMA_20 | 20-day Exponential Moving Average |
| EMA_100 | 100-day Exponential Moving Average |
| RSI_14 | 14-day Relative Strength Index |
| MACD | Difference between 12 and 26 EMA |
| MACD_Signal | 9-day EMA of MACD |
| Bollinger Bands | Upper/Lower bands based on 20-day SMA and standard deviation |
| OBV | On-Balance Volume |
| Daily Return | Percentage daily price change |

**Target Variable:**
```python
Target = 1 if next_day_close > today_close else 0
```

---

## 📊 Data Visualization

We used various plots to understand the market behavior:

- Candlestick Chart + Volume
- RSI and MACD Signals
- OBV vs. Price Line
- Histogram of Daily Returns
- Correlation Heatmap

> These insights helped us validate and visualize feature effectiveness.

---

## ⚙️ Modeling

We used **XGBoost Classifier** with `train_test_split`.

```python
from xgboost import XGBClassifier
```

### Initial Performance

| Metric | Score |
|--------|-------|
| Accuracy | ~0.58 |
| Precision | ~0.63 (Class 1) |
| Recall | ~0.58 |
| F1-Score | ~0.60 |

---

## 🔧 Hyperparameter Tuning

Used `RandomizedSearchCV` and `GridSearchCV` to optimize:

- `n_estimators`
- `max_depth`
- `learning_rate`
- `subsample`
- `colsample_bytree`

➡️ Marginal performance improvement due to data noise and market randomness.

---

## 🧪 Technologies Used

- Python 🐍
- Pandas, NumPy
- Scikit-learn
- XGBoost
- Matplotlib, Seaborn
- Jupyter / Google Colab

---

## 🚀 Future Work

- Integrate **news/sentiment analysis**
- Try **deep learning models** (LSTM, Temporal CNN)
- Incorporate **options data or macro indicators**
- Build a **backtesting trading strategy**

---

## 📦 Installation

```bash
git clone https://github.com/yourusername/stock-direction-xgboost.git
cd stock-direction-xgboost
pip install -r requirements.txt
```

---

## 📁 Project Structure

```
📁 stock-direction-xgboost/
├── data/              # Raw and processed data
├── notebooks/         # EDA and modeling notebooks
├── models/            # Trained models (optional)
├── results/           # Charts, metrics, reports
├── README.md
└── requirements.txt
```

---

## 🧑‍💻 Author

- **Your Name**
- GitHub: [github.com/yourusername](https://github.com/yourusername)
- LinkedIn / Twitter: Optional

---

## 📄 License

This project is licensed under the **MIT License**. See the [LICENSE](./LICENSE) file for more details.

---
