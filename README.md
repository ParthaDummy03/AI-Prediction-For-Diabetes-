#Diabetes risk check for Bangladesh ```
The Model is about the prediction of Diabetes in Bangladesh using AI. Diabetes is a growing public health concern in Bangladesh. Many people remain unaware of their diabetes risk until complications develop. Early identification and timely screening can help reduce the risk of serious health complications.
Diabetes risk is influenced by several factors, including age, gender, blood glucose levels, blood pressure, BMI, insulin levels, skin thickness, and hereditary factors.
However, identifying high-risk individuals is challenging because:
1. Multiple Risk Factors: Diabetes cannot be predicted reliably using a single health measurement. A combination of clinical, physiological, and demographic features is needed to estimate risk.
2. Missing Medical Information: Healthcare datasets frequently contain incomplete patient records. Missing glucose, insulin, BMI, or other clinical measurements can affect prediction accuracy if not handled properly.
3. Geographic and Temporal Analysis: Health records from different districts and months can help demonstrate how a prediction system organizes and analyzes geographically and temporally associated patient data.
4. Early Risk Identification: Without appropriate screening, individuals at elevated risk may not receive timely medical attention.

#Setup in Windows
```bash
1. python -m venv .venv and activate it
2. pip install -r requirements.txt
3. python train.py (writes model/model.joblib and model/metrics.json)
4. python app.py and open http://127.0.0.1:5000
```
#Deployment
gunicorn app:app --bind 0.0.0.0:8000
Commit model/ so the server never retrains. Rerun train.py only when the data changes.

#Notes
The dataset is synthetic. District, Year, and Month are excluded from the model. Scores are uncalibrated.
