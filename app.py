from __future__ import annotations

import re
from typing import List

from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI(title="LLM Talent Matching Engine", version="1.0.0")


class MatchRequest(BaseModel):
    job_description: str
    candidates: List[str]


class CandidateMatch(BaseModel):
    candidate_text: str
    score: float
    matched_keywords: List[str]
    reasoning: str


class MatchResponse(BaseModel):
    ranked_matches: List[CandidateMatch]


TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z+.#-]{1,}")


def keywords(text: str) -> set[str]:
    return {t.lower() for t in TOKEN_RE.findall(text) if len(t) > 2}


def explain_match(job: str, candidate: str, score: float) -> CandidateMatch:
    overlap = sorted(list(keywords(job) & keywords(candidate)))[:12]
    if overlap:
        reasoning = (
            f"Strong overlap in relevant terms such as {', '.join(overlap[:6])}. "
            f"Overall similarity score suggests {'high' if score > 0.5 else 'moderate' if score > 0.25 else 'early'} fit."
        )
    else:
        reasoning = "Limited direct keyword overlap, but semantic similarity still provides a baseline fit estimate."
    return CandidateMatch(candidate_text=candidate, score=round(float(score), 4), matched_keywords=overlap, reasoning=reasoning)


@app.get("/")
def root() -> dict:
    return {"message": "Talent matching API is running."}


@app.post("/match", response_model=MatchResponse)
def match_candidates(payload: MatchRequest) -> MatchResponse:
    documents = [payload.job_description] + payload.candidates
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(documents)
    job_vec = matrix[0:1]
    cand_vecs = matrix[1:]
    scores = cosine_similarity(job_vec, cand_vecs).flatten()

    results = [explain_match(payload.job_description, c, s) for c, s in zip(payload.candidates, scores)]
    results.sort(key=lambda x: x.score, reverse=True)
    return MatchResponse(ranked_matches=results)
