# fairness_boost.py

def fairness_boost(candidate):
    boost = 0
    if candidate["district"].lower() == "rural":
        boost += 0.1
    if candidate["category"].lower() == "sc":
        boost += 0.1
    if candidate["gender"].lower() == "female":
        boost += 0.05
    return boost

candidate = {
    "name": "Aditi Sharma",
    "category": "SC",
    "gender": "Female",
    "district": "Rural"
}

boost_score = fairness_boost(candidate)
print("Candidate:", candidate)
print("Fairness Boost Applied:", boost_score * 100, "% extra")
