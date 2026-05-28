# Customer Support Intelligence Platform

A portfolio-ready data science project for customer support operations. The system cleans support ticket data, analyses SLA performance, predicts SLA breach risk, and provides a polished Streamlit dashboard for decision-making.

## Tech Stack
Python, Pandas, NumPy, Scikit-learn, Plotly, Streamlit, SQL-ready CSV workflow, Power BI-ready processed data.

## Key Features
- Clean and standardise support ticket datasets
- Create SLA target, SLA breach, resolution time, ticket age, and workload features
- Train a Random Forest SLA breach prediction model
- Interactive Streamlit UI with filters, KPIs, charts, risk scores, and recommendations
- Demo dataset included so the app can run immediately

## Project Structure
```text
customer-support-intelligence-platform/
├── app/streamlit_app.py
├── data/raw/support_tickets.csv
├── data/processed/cleaned_tickets.csv
├── src/data_cleaning.py
├── src/feature_engineering.py
├── src/model.py
├── models/sla_breach_model.pkl
├── requirements.txt
└── README.md
```

## Run Locally

### 1. Create and activate environment
```bash
python -m venv venv
venv\Scripts\activate.bat
```

### 2. Install packages
```bash
pip install -r requirements.txt
```

### 3. Clean data
```bash
python src/data_cleaning.py
```

### 4. Train model
```bash
python src/model.py
```

### 5. Run app
```bash
python -m streamlit run app/streamlit_app.py
```

## Resume Description
**Customer Support Intelligence Platform – SLA Risk Prediction and Operational Analytics**  
Tech: Python, Streamlit, scikit-learn, pandas, NumPy, Plotly, Power BI, SQL

Developed an AI-powered operational analytics platform that analyses customer support ticket data, predicts SLA breach risk, and visualises support performance metrics through an interactive dashboard.
Built an end-to-end data pipeline for ticket cleaning, feature engineering, SLA calculation, trend analysis, and ML-based risk scoring.
Implemented a Random Forest model to identify high-risk tickets and generate recommendations such as “Escalate now”, “Monitor closely”, or “Normal handling”.
