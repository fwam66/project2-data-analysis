import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import normalized_mutual_info_score

merged_data = pd.read_csv("merged_data.csv")

merged_data["duration"] = pd.to_numeric(merged_data["duration"],errors="coerce")
merged_data["cumdist"] = pd.to_numeric(merged_data["cumdist"],errors="coerce")

merged_data = merged_data.dropna(subset=["duration","cumdist"])

#continuous variables 
x_cont = merged_data["duration"]
y_cont = merged_data["cumdist"]

#bins for heatmap, adjust if needed
bins = 10

x_bin = pd.qcut(merged_data["duration"], q=bins, duplicates="drop")
y_bin = pd.qcut(merged_data["cumdist"], q=bins, duplicates="drop")

#change bins into numbers for nmi
x_disc = merged_data["duration"].astype("category").cat.codes
y_disc = merged_data["cumdist"].astype("category").cat.codes

plot_df = pd.DataFrame({"duration":x_bin, "cumdist":y_bin})

#pearson correlation
pearson_corr = x_cont.corr(y_cont, method="pearson")
print("Pearson Correlation:", pearson_corr)

#normalized mutual information
nmi = normalized_mutual_info_score(x_disc, y_disc, average_method="min")
print("Normalized Mutual Information:", nmi)

#scatterplot of pearson
plt.figure(figsize=(10,6))
plt.scatter(x_cont, y_cont, alpha=0.5)
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

