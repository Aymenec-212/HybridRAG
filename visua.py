import pandas as pd
import matplotlib.pyplot as plt

# Load results
df = pd.read_csv("topk_eval_results1.csv")

# Compute success rates
hit_rates = {
    "Top-1": df["hit@1"].mean(),
    "Top-3": df["hit@3"].mean(),
    "Top-5": df["hit@5"].mean()
}

# Convert to percentages
hit_rates = {k: v * 100 for k, v in hit_rates.items()}

# Plot
plt.figure(figsize=(8, 6))
plt.bar(hit_rates.keys(), hit_rates.values(), color="skyblue")
plt.ylim(0, 100)
plt.ylabel("Taux de Succès (%)")
plt.xlabel("Top k")
plt.title("")

# Annotate bars with values inside (centered vertically)
for k, v in hit_rates.items():
    plt.text(
        k,                      # x-position
        v / 2,                  # y-position → halfway up the bar
        f"{v:.1f}%",            # text
        ha="center",            # center horizontally
        va="center",            # center vertically
        fontsize=20,
        color="white"           # white looks better on dark bars
    )


plt.show()
