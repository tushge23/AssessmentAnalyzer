# AssessmentAnalyzer

A full-stack Flask web application demonstrating SQL, programming, statistics, and Excel
reporting, with applied machine learning for respondent clustering and dropout-risk
prediction, built around a psychometric and survey-analysis use case.

Built to demonstrate the core application developer skill set: SQL, programming,
statistics, and Excel, delivered as a working full-stack application rather than a
standalone script.

Author: [tushge23](https://github.com/tushge23)

## Project Structure

```
AssessmentAnalyzer/
├── app.py                      # Flask application (routes, upload handling)
├── models/
│   ├── stats_utils.py          # Cronbach's alpha, item difficulty and discrimination
│   ├── ml_utils.py             # K-means clustering and dropout-risk logistic regression
│   └── excel_export.py         # In-memory Excel report generation (openpyxl)
├── templates/
│   ├── base.html
│   ├── index.html              # Upload form and upload history (from SQLite)
│   └── results.html            # Analysis results page
├── static/
│   └── style.css
├── sample_data/
│   └── generate_survey.py      # Generates a realistic sample survey dataset
└── requirements.txt
```

## Quickstart

```bash
pip install -r requirements.txt

# 1. Generate a sample survey dataset to try the app with
python sample_data/generate_survey.py

# 2. Run the app
python app.py
```

Then open http://localhost:5000, upload `sample_data/sample_survey.csv`, and view the analysis.

## What each layer demonstrates

| Skill | Where |
|---|---|
| Programming | `app.py` — Flask routes, file upload handling, state management via `app.config`, error handling |
| SQL | `app.py` (`init_db`, `log_upload`, `get_upload_history`) — SQLite table tracking upload history, queried on the index page |
| Statistics | `models/stats_utils.py` — Cronbach's alpha (reliability), item difficulty, item-total correlation (discrimination), standard psychometric and assessment metrics |
| Machine Learning | `models/ml_utils.py` — K-means clustering of respondents by response pattern; logistic regression predicting survey dropout or non-completion (AUC reported) |
| Excel | `models/excel_export.py` — generates a multi-sheet, styled .xlsx report on demand, streamed as a download with no temp files |

## Design rationale

This mirrors real assessment and measurement work: upload raw item-level response data
and get back reliability and item-quality diagnostics instead of just summary scores,
then export a shareable report. The dropout-risk model applies predictive ML to a
genuinely useful evaluation question — which respondents are at risk of not completing
an assessment.

## Tech Stack

Python, Flask, SQLite, pandas, scikit-learn, scipy, openpyxl

## Testing

The app has an end-to-end path (upload, analyze, view results, download Excel report)
verified via Flask's test client. It can be re-run with:

```python
from app import app, init_db
init_db()
client = app.test_client()
client.get("/")
with open("sample_data/sample_survey.csv", "rb") as f:
    client.post("/analyze", data={"survey_file": (f, "sample_survey.csv")}, content_type="multipart/form-data")
client.get("/download-excel")
```

## License

MIT
