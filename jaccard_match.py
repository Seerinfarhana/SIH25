

def jaccard_similarity(set1, set2):
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union != 0 else 0

student_skills = {"Python", "SQL"}
internship_skills = {"Python", "ML", "SQL"}

score = jaccard_similarity(student_skills, internship_skills)
print("Student Skills:", student_skills)
print("Internship Skills:", internship_skills)
print("Jaccard Match Score:", round(score * 100, 2), "%")
