# cosine_match.py
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

student_text = "Skilled in Python, SQL, Data Analysis"
internship_text = "Looking for interns skilled in Python, Machine Learning, SQL"

emb1 = model.encode(student_text, convert_to_tensor=True)
emb2 = model.encode(internship_text, convert_to_tensor=True)

similarity = util.cos_sim(emb1, emb2).item()
print("Student:", student_text)
print("Internship:", internship_text)
print("Cosine Similarity Match Score:", round(similarity * 100, 2), "%")
