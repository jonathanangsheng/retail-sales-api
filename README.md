# Retail Store Analytics API

Predict customer footfall and optimize pricing strategies using machine learning.

## 🎯 What This API Does

Since the dataset has no "sales" column, we predict **footfall** (customer visits) instead:

**OUTPUT 1:** Predict how many customers will visit based on:
- Price, discounts, promotions
- Weather conditions
- Customer sentiment
- Competition

**OUTPUT 2:** Find the optimal price that maximizes customer footfall

## 🚀 Quick Start

1. **Download dataset** from Kaggle and save as `retail_sales.csv`

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run API:**
```bash
uvicorn main:app --reload
```

4. **Open browser:**
