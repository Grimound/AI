import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
file_path = "https://raw.githubusercontent.com/Grimound/AI/refs/heads/main/SATandGPA.csv"
df = pd.read_csv(file_path)
GPA = df["GPA"]
SAT_Score = df["SAT Score"]
plt.figure(figsize=(8, 5))
plt.scatter(GPA, SAT_Score)
plt.xlabel("GPA")
plt.ylabel("SAT Score")
plt.title("Relationship between GPA and SAT scores")
plt.grid(True)
plt.show()
a, b = np.polyfit(GPA, SAT_Score, 1)
plt.plot(df["GPA"], a * df["GPA"] + b, color="red", label="Regression Line")
plt.scatter(df["GPA"], df["SAT Score"])
plt.xlabel("GPA")
plt.ylabel("SAT Score")
plt.title("Relationship between GPA and SAT scores")
plt.legend()
plt.grid(True)
plt.show()
test_SAT = 1100
test_GPA = (test_SAT - b) / a
print(f"Predicted GPA for SAT score of 1100: {test_GPA:.2f}")
random_SAT = np.random.randint(400, 1601)
random_GPA = (random_SAT - b) / a
print(f"Predicted GPA for random SAT score of {random_SAT}: {random_GPA:.2f}")
