# LLM Talent Matching Engine

A recruiter-focused matching API that ranks candidate profiles against a job description.

## What it does
- Scores profile/job similarity with TF-IDF + cosine similarity
- Returns ranked candidates
- Produces lightweight human-readable reasoning and matched keywords

## Why it matters
This project demonstrates ranking systems, feature engineering, explainability, and API delivery.

## Run locally
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload
```

## Example request
```json
{
  "job_description": "Need a machine learning engineer with Python, FastAPI, AWS, NLP and React experience.",
  "candidates": [
    "ML engineer with Python, AWS, FastAPI and NLP project experience.",
    "Frontend developer focused on CSS, Figma and design systems.",
    "Backend engineer with Python, APIs, Docker and cloud deployment experience."
  ]
}
```
