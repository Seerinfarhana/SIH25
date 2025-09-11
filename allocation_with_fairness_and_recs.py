"""
Allocation engine with:
- Fairness boost (rule-based)
- Alternative recommendations (top-N internships per candidate) via embeddings + KNN

Save as: allocation_with_fairness_and_recs.py
Run: python allocation_with_fairness_and_recs.py
"""

from typing import List, Dict, Any, Tuple
import numpy as np
from collections import defaultdict

# Try to import SBERT, otherwise fallback to TF-IDF vectors
TRY_SBERT = True
try:
    from sentence_transformers import SentenceTransformer, util
    model = SentenceTransformer("all-MiniLM-L6-v2")
    USE_SBERT = True
except Exception as e:
    USE_SBERT = False
    print("Warning: sentence-transformers not available or failed to load. Falling back to TF-IDF method.")
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    tfidf_vectorizer = TfidfVectorizer()

# ---------- Sample Data ----------
candidates = [
    {
        "id": "U001",
        "name": "Aditi Sharma",
        "skills": ["Python", "SQL", "Data Analysis"],
        "category": "SC",
        "gender": "Female",
        "district": "Rural",
        "preferences": ["Remote", "AI"]
    },
    {
        "id": "U002",
        "name": "Rahul Mehta",
        "skills": ["Java", "Spring", "SQL"],
        "category": "GEN",
        "gender": "Male",
        "district": "Urban",
        "preferences": ["Onsite", "Backend"]
    },
    {
        "id": "U003",
        "name": "Neha Verma",
        "skills": ["Python", "Data Visualization"],
        "category": "OBC",
        "gender": "Female",
        "district": "Rural",
        "preferences": ["Remote", "Data"]
    }
]

internships = [
    {
        "id": "I001",
        "title": "AI Research Intern",
        "skills_required": ["Python", "ML", "Data Analysis"],
        "capacity": 2,
        "quotas": {"rural": 1, "sc": 1},  # keys map to candidate attributes (see apply_quotas)
        "location": "Remote",
        "description": "Research tasks in Python, Machine Learning and Data Analysis"
    },
    {
        "id": "I002",
        "title": "Backend Developer Intern",
        "skills_required": ["Java", "SQL"],
        "capacity": 1,
        "quotas": {},
        "location": "Onsite",
        "description": "Backend APIs using Java and SQL"
    },
    {
        "id": "I003",
        "title": "Data Visualization Intern",
        "skills_required": ["Python", "Visualization", "Tableau"],
        "capacity": 1,
        "quotas": {"female": 1},
        "location": "Remote",
        "description": "Create dashboards and visualization using Python and Tableau"
    }
]

# ---------- Utilities ----------
def text_from_skills(skills: List[str]) -> str:
    return " ".join(skills)

def jaccard_similarity(set1: set, set2: set) -> float:
    inter = len(set1 & set2)
    union = len(set1 | set2)
    return inter / union if union != 0 else 0.0

# ---------- Embedding helpers ----------
def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Return embeddings for list of texts.
    Falls back to TF-IDF vectors if SBERT isn't available.
    """
    if USE_SBERT:
        embs = model.encode(texts, convert_to_tensor=False, show_progress_bar=False)
        return np.array(embs)
    else:
        X = tfidf_vectorizer.fit_transform(texts)
        return X.toarray()

def cosine_sim_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    # compute cosine similarities between rows of A and rows of B
    # A: n x d, B: m x d -> result n x m
    if A.size == 0 or B.size == 0:
        return np.zeros((A.shape[0], B.shape[0]))
    # normalize
    An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
    Bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-9)
    return An @ Bn.T

# ---------- Fairness boost (configurable) ----------
def fairness_boost(candidate: Dict[str, Any], boost_config: Dict[str, float]) -> float:
    """
    Compute additive boost fraction (0..1) for candidate based on boost_config.
    Example boost_config = {"rural": 0.10, "sc": 0.10, "female": 0.05}
    Candidate attributes: district, category, gender
    """
    boost = 0.0
    if boost_config.get("rural") and candidate.get("district", "").lower() == "rural":
        boost += boost_config["rural"]
    if boost_config.get("sc") and candidate.get("category", "").lower() in ("sc", "st"):
        boost += boost_config["sc"]
    if boost_config.get("female") and candidate.get("gender", "").lower() == "female":
        boost += boost_config["female"]
    return boost

# ---------- Scoring ----------
def compute_score(candidate: Dict[str, Any],
                  internship: Dict[str, Any],
                  cand_emb: np.ndarray = None,
                  intern_emb: np.ndarray = None,
                  weights: Dict[str, float] = None,
                  boost_config: Dict[str, float] = None) -> Tuple[float, dict]:
    """
    Compute hybrid final score as percentage (0..100).
    weights: dict with keys cosine, jaccard, loc, boost
    """
    if weights is None:
        weights = {"cosine": 0.6, "jaccard": 0.2, "loc": 0.1, "boost": 0.1}
    if boost_config is None:
        boost_config = {"rural": 0.1, "sc": 0.1, "female": 0.05}
    # cosine similarity from embeddings (if provided)
    cos_sim = 0.0
    if cand_emb is not None and intern_emb is not None:
        # cand_emb and intern_emb are 1-D vectors
        num = np.dot(cand_emb, intern_emb)
        den = (np.linalg.norm(cand_emb) * np.linalg.norm(intern_emb) + 1e-9)
        cos_sim = float(num / den)
        # clamp 0..1
        cos_sim = max(0.0, min(1.0, cos_sim))
    # jaccard on skills
    jaccard = jaccard_similarity(set([s.lower() for s in candidate.get("skills", [])]),
                                 set([s.lower() for s in internship.get("skills_required", [])]))
    # location preference
    loc_pref = 1.0 if internship.get("location") in candidate.get("preferences", []) else 0.0
    # fairness boost fraction
    boost_fraction = fairness_boost(candidate, boost_config)
    # Weighted sum
    final = (weights["cosine"] * cos_sim +
             weights["jaccard"] * jaccard +
             weights["loc"] * loc_pref +
             weights["boost"] * boost_fraction)
    return round(final * 100, 2), {
        "cosine": round(cos_sim, 3),
        "jaccard": round(jaccard, 3),
        "loc_pref": int(loc_pref),
        "boost_fraction": round(boost_fraction, 3)
    }

# ---------- Allocation with quota enforcement ----------
def allocate_with_quotas(candidates: List[Dict], internships: List[Dict],
                         weights=None, boost_config=None) -> Dict[str, Any]:
    """
    For each internship:
      - compute score for all candidates
      - apply quotas (internship['quotas'] is dict e.g. {'rural':1, 'sc':1, 'female':1})
      - fill remaining seats by highest score
    Returns allocations dict keyed by internship id.
    """
    # Precompute embeddings (for SBERT or TF-IDF flow)
    cand_texts = [text_from_skills(c["skills"]) for c in candidates]
    intern_texts = [intern["description"] if intern.get("description") else text_from_skills(intern["skills_required"]) for intern in internships]
    cand_embs = embed_texts(cand_texts)
    intern_embs = embed_texts(intern_texts)

    allocations = {}
    for i_idx, intern in enumerate(internships):
        scores = []
        for c_idx, cand in enumerate(candidates):
            score_pct, breakdown = compute_score(cand, intern,
                                                 cand_emb=cand_embs[c_idx],
                                                 intern_emb=intern_embs[i_idx],
                                                 weights=weights,
                                                 boost_config=boost_config)
            missing = list(set(intern.get("skills_required", [])) - set(cand.get("skills", [])))
            scores.append({
                "candidate_index": c_idx,
                "candidate_id": cand["id"],
                "name": cand["name"],
                "score": score_pct,
                "breakdown": breakdown,
                "missing_skills": missing,
                "category": cand.get("category", ""),
                "district": cand.get("district", ""),
                "gender": cand.get("gender", "")
            })

        # sort by score descending
        scores.sort(key=lambda x: x["score"], reverse=True)

        selected = []
        selected_ids = set()

        # apply quotas
        quotas = intern.get("quotas", {})  # e.g., {'rural':1, 'sc':1}
        for q_key, q_num in quotas.items():
            # find candidates satisfying this quota and not already selected
            # mapping quota key to candidate attribute check:
            # - 'rural' -> district == 'Rural'
            # - 'sc' -> category in ['SC','ST']
            # - 'female' -> gender == 'Female'
            eligible = []
            for s in scores:
                if s["candidate_id"] in selected_ids:
                    continue
                if q_key.lower() == "rural" and s["district"].lower() == "rural":
                    eligible.append(s)
                elif q_key.lower() in ("sc", "st") and s["category"].lower() in ("sc", "st"):
                    eligible.append(s)
                elif q_key.lower() == "female" and s["gender"].lower() == "female":
                    eligible.append(s)
                # add other quota keys mapping as needed
            # pick top q_num from eligible
            for e in eligible[:q_num]:
                selected.append(e)
                selected_ids.add(e["candidate_id"])

        # fill remaining capacity with highest-scoring not-yet-selected
        capacity = intern.get("capacity", 1)
        for s in scores:
            if len(selected) >= capacity:
                break
            if s["candidate_id"] not in selected_ids:
                selected.append(s)
                selected_ids.add(s["candidate_id"])

        # store
        allocations[intern["id"]] = {
            "title": intern.get("title"),
            "allocated": selected,
            "all_scores_sorted": scores
        }
    return allocations

# ---------- Alternative Recommendations (top-N internships per candidate) ----------
def recommendations_top_n(candidates: List[Dict], internships: List[Dict], top_n: int = 3) -> Dict[str, List[Tuple[str, float]]]:
    """
    For each candidate, returns top_n internships by embedding cosine similarity (semantic).
    Returns dict mapping candidate_id -> list of (internship_id, similarity_score_pct).
    """
    cand_texts = [text_from_skills(c["skills"]) for c in candidates]
    intern_texts = [intern["description"] if intern.get("description") else text_from_skills(intern["skills_required"]) for intern in internships]
    cand_embs = embed_texts(cand_texts)
    intern_embs = embed_texts(intern_texts)

    sim_matrix = cosine_sim_matrix(cand_embs, intern_embs)  # n_candidates x n_internships
    recs = {}
    for i, cand in enumerate(candidates):
        sims = sim_matrix[i]  # array over internships
        idx_sorted = np.argsort(-sims)  # descending
        top = []
        for idx in idx_sorted[:top_n]:
            top.append((internships[idx]["id"], float(round(sims[idx] * 100, 2))))
        recs[cand["id"]] = top
    return recs

# ---------- Demo run ----------
if __name__ == "__main__":
    # configure weights and boost
    weights = {"cosine": 0.6, "jaccard": 0.2, "loc": 0.1, "boost": 0.1}
    boost_config = {"rural": 0.10, "sc": 0.10, "female": 0.05}

    print("=== Running allocation with quotas and fairness boosts ===")
    allocations = allocate_with_quotas(candidates, internships, weights=weights, boost_config=boost_config)
    import json
    print(json.dumps(allocations, indent=2))

    print("\n=== Running alternative recommendations (top-3) ===")
    recs = recommendations_top_n(candidates, internships, top_n=3)
    print(json.dumps(recs, indent=2))
