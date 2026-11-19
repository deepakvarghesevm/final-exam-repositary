# Flask ML Prediction Application

This project serves `adult_income_pipeline.pkl` through a local Flask web application.

## Windows VS Code terminal

Open the project folder:

```powershell
cd "C:\Users\LAB-USER-01\Desktop\DEEPAK VARGHESE\Flask_Project"
code .
```

Create and activate a virtual environment:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the application:

```powershell
python app.py
```

Open `http://127.0.0.1:5000` in your browser. The terminal prints a different URL if port 5000 is already occupied.

To request another starting port explicitly:

```powershell
$env:FLASK_PORT="5001"
python app.py
```

Stop the server with `Ctrl+C`. Debug mode and the automatic reloader are disabled.

## Model details

- Model file: `adult_income_pipeline.pkl`
- Problem type: regression
- Final estimator: `DecisionTreeRegressor`
- Pipeline feature order:
  1. `company_type`
  2. `employees`
  3. `employees_year`
  4. `revenue_year`
  5. `net_income`
  6. `net_income_year`
  7. `legal_units_count`
  8. `direct_subsidiaries_count`
  9. `max_hierarchy_depth`
  10. `is_standalone_entity`
  11. `revenue_usd`

The pipeline contains a numeric `StandardScaler`, a categorical `OneHotEncoder` for `company_type`, and passthrough handling for `is_standalone_entity`. Valid `company_type` choices are detected from the fitted encoder:

`NONPROFIT`, `OTHER`, `PRIVATE`, `PUBLIC`, `Private`, `Public`, `STATE_OWNED`, `avolon.aero`.
