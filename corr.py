import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import normalized_mutual_info_score

merged_data = pd.read_csv("merged_data.csv")

merged_data["duration"] = pd.to_numeric(merged_data["duration"],errors="coerce")
merged_data["cumdist"] = pd.to_numeric(merged_data["cumdist"],errors="coerce")

merged_data = merged_data.dropna(subset=["duration","cumdist"])

#bins for heatmap, adjust if needed
bins = 10

x_bin = pd.qcut(merged_data["duration"], q=bins, duplicates="drop")
y_bin = pd.qcut(merged_data["cumdist"], q=bins, duplicates="drop")

#change categorical data into numbers
x = merged_data["duration"].astype("category").cat.codes
y = merged_data["cumdist"].astype("category").cat.codes

plot_df = pd.DataFrame({"duration":x_bin, "cumdist":y_bin})

#pearson correlation
pearson_corr = x.corr(y, method="pearson")
print("Pearson Correlation:", pearson_corr)

#normalized mutual information
nmi = normalized_mutual_info_score(x, y, average_method="min")
print("Normalized Mutual Information:", nmi)

#scatterplot of pearson - looks like a mess bcz data is discrete
plt.figure(figsize=(10,6))
plt.scatter(x, y, alpha=0.5)
plt.title("Scatterplot of Duration and Cummulative Distance")
plt.xlabel("Duration")
plt.ylabel("Cummulative Distance")
plt.show()

#heatmap of nmi
plt.figure(figsize=(10,6))
heatmap = pd.crosstab(plot_df["duration"], plot_df["cumdist"])
sns.heatmap(heatmap, annot=True, fmt="d")
plt.title("Heatmap of Duration and Cummulative Distance")
plt.xlabel("Duration")
plt.ylabel("Cummulative Distance")
plt.show()
