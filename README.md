# Bitcoin Forecasting Web App (Permutation Decision Trees)

A Flask-based web application for Bitcoin time-series forecasting experiments.  
The project focuses on **Permutation Decision Trees (PDT)** for short-horizon forecasting and provides an interactive UI to:

- fetch and visualize BTC candle data (OHLCV)
- build features (incl. Ichimoku-derived signals)
- run a forecasting experiment (classification or regression)
- display forecast output, confidence, metrics, and a visualization dashboard

> Current version includes a lightweight PDT-style implementation as a permutation-bootstrapped tree ensemble and a Decision Tree baseline.

---

## Tech Stack

- **Backend:** Python, Flask  
- **Data:** pandas, numpy  
- **ML:** scikit-learn  
- **Charts:** matplotlib, mplfinance  
- **API / Data source:** `t-tech-investments` (T-Bank Invest API SDK)

---

## Setup

### Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
If you install the T-Bank SDK from the official package index:
python -m pip install "t-tech-investments==0.3.3" \
  --index-url https://opensource.tbank.ru/api/v4/projects/238/packages/pypi/simple
python app.py