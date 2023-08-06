import mplcyberpunk
import pandas as pd
import matplotlib.pyplot as plt

# Styling for matplotlib
plt.style.use("cyberpunk")
plt.style.use("dark_background")

meanScores = pd.read_csv('sentiment_mean_scores.csv')

plt.figure(figsize=(18, 10))
plt.bar(meanScores.index, meanScores['compound'], color="#DAFF00")
plt.xlabel('Equity')
plt.ylabel('Average Sentiment Score')
plt.title('Average Sentiment Score per Equity')
plt.xticks(ticks=meanScores.index, labels=meanScores['Equity'], rotation=45)  # Set x-ticks to equity names
plt.grid(False)
mplcyberpunk.add_glow_effects()
plt.tight_layout()
plt.savefig('sentiment_mean.png')
plt.show()
