
import re

# Define a dictionary of common skills
SKILL_DICTIONARY = ["Python", "SQL", "Java", "ML", "Data Analysis", "Communication", "Spring", "AI"]

def extract_skills(text):
    skills_found = []
    for skill in SKILL_DICTIONARY:
        if re.search(rf"\b{skill}\b", text, re.IGNORECASE):
            skills_found.append(skill)
    return skills_found

# Example resume
resume_text = "I have experience in Python, SQL and strong Communication skills."
print("Resume:", resume_text)
print("Extracted Skills:", extract_skills(resume_text))
