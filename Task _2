import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

print("Enter spending scores (1-100) for 6 buyers:")

spending_scores = []
for i in range(6):
    val = float(input(f"Buyer {i+1} spending score: "))
    spending_scores.append(val)

data = pd.DataFrame({'Buyer': [f"Buyer {i+1}" for i in range(6)],
                     'Spending Score (1-100)': spending_scores})

X = data[['Spending Score (1-100)']]
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
data['Cluster'] = kmeans.fit_predict(X)

cluster_means = data.groupby('Cluster')['Spending Score (1-100)'].mean().sort_values()
mapping = {cluster: idx for idx, cluster in enumerate(cluster_means.index)}
categories = {0: "Low Spender", 1: "Medium Spender", 2: "High Spender"}

data['Category'] = data['Cluster'].map(mapping).map(categories)

print("\n=== Buyer Categories ===")
for i, row in data.iterrows():
    print(f"{row['Buyer']}: Spending Score = {row['Spending Score (1-100)']}, Category = {row['Category']}")

plt.figure(figsize=(8,5))
colors = {"Low Spender": "red", "Medium Spender": "orange", "High Spender": "green"}
plt.bar(data['Buyer'], data['Spending Score (1-100)'],
        color=[colors[cat] for cat in data['Category']])

plt.title("Customer Segmentation by Spending Score")
plt.xlabel("Buyers")
plt.ylabel("Spending Score (1-100)")
plt.legend(colors, title="Category")
plt.show()
