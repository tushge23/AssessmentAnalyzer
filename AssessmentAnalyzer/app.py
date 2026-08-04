"""
app.py
Flask web app: AssessmentAnalyzer

Upload a CSV of Likert-scale survey responses (columns: item_1..item_n, optionally
'completed' and 'response_time_sec') and get back:
 - Reliability analysis (Cronbach's alpha)
 - Item difficulty / discrimination
 - K-means respondent clustering
 - Optional dropout-risk prediction (if a 'completed' column is present)
 - Downloadable formatted Excel report

Run:
    python app.py
Then open http://localhost:5000
"""

import io
import sqlite3
import os

import pandas as pd
from flask import Flask, render_template, request, send_file, redirect, url_for, flash

from models.stats_utils import full_report
from models.ml_utils import cluster_respondents, predict_dropout_risk
from models.excel_export import build_report

app = Flask(__name__)
app.secret_key = "dev-secret-key-change-in-production"

DB_PATH = os.path.join(os.path.dirname(__file__), "surveys.db")
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS uploads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            n_respondents INTEGER,
            n_items INTEGER,
            cronbach_alpha REAL,
            uploaded_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


def log_upload(filename, n_respondents, n_items, alpha):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO uploads (filename, n_respondents, n_items, cronbach_alpha) VALUES (?, ?, ?, ?)",
        (filename, n_respondents, n_items, alpha),
    )
    conn.commit()
    conn.close()


def get_upload_history():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT filename, n_respondents, n_items, cronbach_alpha, uploaded_at "
        "FROM uploads ORDER BY id DESC LIMIT 10"
    ).fetchall()
    conn.close()
    return rows


@app.route("/", methods=["GET"])
def index():
    history = get_upload_history()
    return render_template("index.html", history=history)


@app.route("/analyze", methods=["POST"])
def analyze():
    file = request.files.get("survey_file")
    if not file or file.filename == "":
        flash("Please choose a CSV file to upload.")
        return redirect(url_for("index"))

    try:
        df = pd.read_csv(file)
    except Exception as e:
        flash(f"Could not read CSV: {e}")
        return redirect(url_for("index"))

    item_cols = [c for c in df.columns if c.startswith("item_")]
    if len(item_cols) < 2:
        flash("CSV must contain at least two columns named item_1, item_2, ... (Likert responses).")
        return redirect(url_for("index"))

    item_df = df[item_cols].apply(pd.to_numeric, errors="coerce")
    scale_max = float(item_df.max().max())

    report = full_report(item_df, scale_max=scale_max)
    cluster_info = cluster_respondents(item_df, n_clusters=3)

    dropout_result = None
    if "completed" in df.columns:
        dropout_result = predict_dropout_risk(df, item_cols, target_col="completed")

    # Save uploaded file + log to DB
    save_path = os.path.join(UPLOAD_DIR, file.filename)
    df.to_csv(save_path, index=False)
    log_upload(file.filename, report["n_respondents"], report["n_items"], report["cronbach_alpha"])

    # Persist last analysis in-memory for the Excel export route (simple demo approach)
    app.config["LAST_ITEM_DF"] = item_df
    app.config["LAST_REPORT"] = report
    app.config["LAST_CLUSTER"] = cluster_info

    return render_template(
        "results.html",
        report=report,
        cluster_info=cluster_info,
        dropout_result=dropout_result,
        filename=file.filename,
    )


@app.route("/download-excel")
def download_excel():
    item_df = app.config.get("LAST_ITEM_DF")
    report = app.config.get("LAST_REPORT")
    cluster_info = app.config.get("LAST_CLUSTER")

    if item_df is None or report is None:
        flash("Run an analysis first before downloading the report.")
        return redirect(url_for("index"))

    excel_bytes = build_report(item_df, report, cluster_info)
    return send_file(
        io.BytesIO(excel_bytes),
        as_attachment=True,
        download_name="assessment_report.xlsx",
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


if __name__ == "__main__":
    init_db()
    app.run(debug=True, port=5000)
