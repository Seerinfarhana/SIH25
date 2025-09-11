# recommend_internships.py
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

student_profile = "Python, Data Analysis"
internships = {
    "AI Research": "Python, ML, Data Analysis",
    "Backend Dev": "Java, SQL, Spring",
    "Data Analyst": "Python, SQL, Visualization",
    "Web Developer": "HTML, CSS, JavaScript"
}

emb_student = model.encode(student_profile, convert_to_tensor=True)

scores = {}
for title, desc in internships.items():
    emb_job = model.encode(desc, convert_to_tensor=True)
    similarity = util.cos_sim(emb_student, emb_job).item()
    scores[title] = similarity

# Sort internships by similarity
sorted_jobs = sorted(scores.items(), key=lambda x: x[1], reverse=True)

print("Student Profile:", student_profile)
print("Top 2 Recommended Internships:")
for job, score in sorted_jobs[:2]:
    print(f"{job} → {round(score*100,2)}% match")
