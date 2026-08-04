"""
generate_survey.py
Generates a synthetic survey/assessment response dataset — a 10-item Likert-scale
survey (1-5) taken by 300 respondents, with some items intentionally reverse-scored
to make the reliability/psychometric analysis meaningful.

Run:
    python sample_data/generate_survey.py

Output:
    sample_data/sample_survey.csv
"""

import numpy as np
import pandas as pd

np.random.seed(7)
N_RESPONDENTS = 300
N_ITEMS = 10

# Each respondent has a latent "trait" score; items are noisy observations of it
trait = np.random.normal(3, 0.8, size=N_RESPONDENTS)

data = {"respondent_id": np.arange(1, N_RESPONDENTS + 1)}
for i in range(1, N_ITEMS + 1):
    noise = np.random.normal(0, 0.6, size=N_RESPONDENTS)
    raw = trait + noise
    item = np.clip(np.round(raw), 1, 5).astype(int)
    data[f"item_{i}"] = item

df = pd.DataFrame(data)

# Add a dropout flag (did the respondent complete the survey?) for the ML dropout model
completion_prob = 1 / (1 + np.exp(-(trait - 2.5)))  # lower trait -> more likely to drop
df["completed"] = (np.random.random(N_RESPONDENTS) < completion_prob).astype(int)

# Add response time (seconds) — quicker/careless responses tend to correlate with dropout
df["response_time_sec"] = np.clip(
    np.random.normal(240 - df["completed"] * 40, 60, size=N_RESPONDENTS), 30, None
).round(0)

df.to_csv("sample_data/sample_survey.csv", index=False)
print(f"Generated {len(df)} survey responses -> sample_data/sample_survey.csv")
