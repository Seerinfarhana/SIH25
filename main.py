from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any
import allocation_with_fairness_and_recs as engine

app = FastAPI(title="Internship Allocation API")

# ---------- Request Models ----------
class Candidate(BaseModel):
    id: str
    name: str
    skills: List[str]
    category: str
    gender: str
    district: str
    preferences: List[str]

class Internship(BaseModel):
    id: str
    title: str
    skills_required: List[str]
    capacity: int
    quotas: Dict[str, int]
    location: str
    description: str = ""

# ---------- API Endpoints ----------
@app.post("/allocate")
def allocate(candidates: List[Candidate], internships: List[Internship]):
    results = engine.allocate_with_quotas(
        [c.dict() for c in candidates],
        [i.dict() for i in internships]
    )
    return results

@app.post("/recommend")
def recommend(candidates: List[Candidate], internships: List[Internship]):
    results = engine.recommendations_top_n(
        [c.dict() for c in candidates],
        [i.dict() for i in internships],
        top_n=3
    )
    return results

