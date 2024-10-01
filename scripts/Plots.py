import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

with open('../configs/results.json', 'r') as f:
    para = json.load(f)

df = pd.DataFrame(para).T

plt.figure(figsize=(10, 8))
sns.heatmap(df, annot=True, cmap='crest_r', linewidths=0.5)
plt.title('Model Performance')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
df['F1-score positive class'].plot(kind='barh', color='skyblue')
plt.title('F1-Score (Positive Class) Comparison Across Models')
plt.xlabel('F1-Score Positive Class')
plt.tight_layout()
plt.show()
