import matplotlib.pyplot as plt

# Distribution values provided by user
rank_distribution = {
    "1": 29,
    "2": 12,
    "3": 2,
    "4": 4,
    "5": 2,
}

# Plot
plt.figure(figsize=(8, 6))
bars = plt.bar(rank_distribution.keys(), rank_distribution.values(), color="skyblue")

# Add annotations inside bars
for bar in bars:
    height = bar.get_height()
    if height > 0:
        plt.text(bar.get_x() + bar.get_width() / 2,
                 height / 2,
                 str(height),
                 ha="center", va="center", fontsize=10, color="black")

plt.title("")
plt.xlabel("Rang du document (Position)")
plt.ylabel("Nombre de requêtes")
plt.tight_layout()
plt.show()
