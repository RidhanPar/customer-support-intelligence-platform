# Customer Support Intelligence Platform

A data analytics and machine learning platform for monitoring support operations, predicting SLA breach risk, identifying high-risk tickets, and helping support teams prioritize work using KPI dashboards and risk-based recommendations.

Live Demo: https://customer-support-intelligence-platform.streamlit.app/

## Dashboard Preview

![Dashboard Overview](assets/screenshots/01_dashboard_overview.png)

## Business Problem

Customer support teams often manage high volumes of tickets across different priorities, channels, categories, and teams. Without a clear way to monitor SLA performance and identify early risk signals, teams may miss tickets that are likely to breach SLA.

This project simulates how a support operations or service management team can use data analytics and machine learning to:

* Monitor SLA performance
* Identify high-risk tickets
* Detect operational bottlenecks
* Prioritize urgent work
* Support faster decision-making for team leads and managers

## Solution Overview

The Customer Support Intelligence Platform provides an interactive Streamlit dashboard that combines operational analytics, SLA risk prediction, ticket-level scoring, and recommended actions.

Users can explore support ticket data, apply business filters, review KPI trends, monitor predicted SLA breach risk, and export filtered results for further analysis.

## Key Features

* Executive dashboard for support operations monitoring
* SLA breach prediction using machine learning
* Ticket-level risk scoring
* Dynamic risk level classification
* Recommended action groups based on predicted risk
* Filters for priority, category, channel, team, and risk level
* Dataset explorer for ticket-level investigation
* Downloadable CSV output for filtered records and prediction results
* Demo dataset included for public portfolio presentation

## Dashboard Sections

### 1. Executive Dashboard

The executive dashboard provides a high-level view of support operations performance, including:

* Total tickets
* SLA breach rate
* Average resolution hours
* Escalation rate
* SLA breach by priority
* Breach rate by category

This section is designed for quick operational monitoring and leadership-level visibility.

### 2. Risk Monitor

The risk monitor focuses on ticket-level SLA breach prediction and action prioritization.

It includes:

* Average predicted risk
* Critical, high, and medium risk ticket counts
* Risk level distribution
* Recommended action mix
* Ticket-level prediction output
* Risk-based filtering
* Downloadable prediction results

This section helps support teams identify which tickets require urgent review, priority handling, monitoring, or normal handling.

### 3. Dataset Explorer

The dataset explorer allows users to inspect the underlying ticket data.

It includes:

* Search by ticket ID, category, team, or status
* Filtered table view
* Raw ticket-level fields
* Downloadable filtered dataset

This section supports deeper investigation and data validation.

## Machine Learning Approach

The project uses a Random Forest classification model to predict the probability of an SLA breach based on operational ticket features.

Example input features include:

* Ticket priority
* Ticket category
* Support channel
* Assigned team
* Ticket age
* Escalation status
* Resolution hours
* Customer satisfaction score

The model output is converted into:

* Predicted SLA breach risk score
* Risk level classification
* Recommended action group

## Model Performance

The model is evaluated using standard classification metrics such as accuracy, precision, recall, F1-score, and ROC AUC.

The current version uses a fictional demo dataset created for portfolio and testing purposes. Because the dataset is controlled and simplified, the model performance should be interpreted as a demonstration of the machine learning workflow rather than a production-level benchmark.

## Business Value

This project demonstrates how support operations teams can use analytics and machine learning to improve service visibility and ticket prioritization.

The platform supports:

* Faster identification of SLA risk
* Better prioritization of high-risk tickets
* Improved operational reporting
* Clearer visibility for team leads and managers
* More structured support decision-making

## Tech Stack

* Python
* Streamlit
* Pandas
* NumPy
* Scikit-learn
* Plotly
* Joblib
* SQL-ready CSV workflow
* Power BI-ready processed data

## Project Structure

```text
customer-support-intelligence-platform/
├── app/
│   └── streamlit_app.py
├── assets/
│   └── screenshots/
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── notebooks/
├── src/
│   ├── data_cleaning.py
│   ├── feature_engineering.py
│   └── model.py
├── requirements.txt
└── README.md
```

## How to Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/RidhanPar/customer-support-intelligence-platform.git
cd customer-support-intelligence-platform
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the virtual environment

For Windows:

```bash
venv\Scripts\activate
```

For macOS or Linux:

```bash
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the app

```bash
streamlit run app/streamlit_app.py
```

## Demo Data

The project includes fictional demo support ticket data for public portfolio use.

Users can either:

* Use the built-in demo dataset
* Upload their own compatible CSV file

No real customer, company, or confidential operational data is included.

## Example Use Case

A support operations manager wants to identify which tickets are most likely to breach SLA.

Using this dashboard, the manager can:

1. Review overall SLA performance.
2. Filter tickets by priority, category, channel, or team.
3. Open the risk monitor.
4. Identify tickets classified as critical or high risk.
5. Review recommended actions.
6. Export prediction results.
7. Prioritize follow-up work for the support team.

## Limitations

This project is built using fictional demo data and is intended for portfolio demonstration.

Current limitations:

* The dataset is simplified and does not represent a real production ticketing system.
* Model performance may be higher than expected because the demo data is controlled.
* The current version does not connect to a live CRM or ticketing database.
* Role-based access control is not yet implemented.
* Automated alerting is planned but not yet included.

## Future Enhancements

Planned improvements include:

* Add model performance metrics directly inside the dashboard
* Add feature importance and explainability
* Add SLA trend forecasting
* Add aging ticket alerts
* Add team-level performance comparison
* Add automated weekly report export
* Connect the dashboard to PostgreSQL
* Add role-based access for managers and analysts
* Add workflow automation integration for escalation alerts

## Portfolio Summary

Customer Support Intelligence Platform is a support operations analytics project that combines SLA monitoring, machine learning risk prediction, interactive dashboards, and action recommendations.

It demonstrates practical skills in data analytics, business intelligence, machine learning, support analytics, dashboard development, and data-driven operational decision-making.
