Diabetes risk check for Bangladesh

Setup
1. python -m venv .venv and activate it
2. pip install -r requirements.txt
3. python train.py (writes model/model.joblib and model/metrics.json)
4. python app.py and open http://127.0.0.1:5000

Deployment
gunicorn app:app --bind 0.0.0.0:8000
Commit model/ so the server never retrains. Rerun train.py only when the data changes.

Notes
The dataset is synthetic. District, Year, and Month are excluded from the model. Scores are uncalibrated.
