import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import normalized_mutual_info_score


#load and merge all the data
merged_data = pd.read_csv("merged_data.csv")

#categorical data
merged_data["weekly_hhinc_group"] = merged_data["weekly_hhinc_group"].astype("category")
merged_data["homelga"] = merged_data["homelga"].astype("category")
merged_data["trippurp"] = merged_data["trippurp"].astype("category")
merged_data["mainmode"] = merged_data["mainmode"].astype("category")


#numerical data
merged_data["totalvehs"] = pd.to_numeric(merged_data["totalvehs"], errors="coerce")
merged_data["triptime"] = pd.to_numeric(merged_data["triptime"], errors="coerce")
merged_data["cumdist"] = pd.to_numeric(merged_data["cumdist"], errors="coerce")

#remove rows with missing values
merged_data = merged_data.dropna(subset=["weekly_hhinc_group","homelga","mainmode",
                                         "trippurp", "totalvehs", "triptime", "cumdist"])


#compute the weighted means - income
#household income is fine because its ordinal

income_mean = []
for group in merged_data["weekly_hhinc_group"].unique():
    column = merged_data[merged_data["weekly_hhinc_group"] == group]
    weighted_vehs = (column["totalvehs"] * column["hhpoststratweight"]).sum()/ column["hhpoststratweight"].sum()
    weighted_dur = (column["triptime"] * column["trippoststratweight"]).sum()/ column["trippoststratweight"].sum()
    weighted_dist = (column["cumdist"] * column["trippoststratweight"]).sum()/ column["trippoststratweight"].sum()
    income_mean.append({
        "income_groups": group,
        "weighted_totalvehs": weighted_vehs,
        "weighted_triptime": weighted_dur,
        "weighted_cumdist": weighted_dist
        })

income_mean = pd.DataFrame(income_mean)
income_mean["income_groups"] = income_mean["income_groups"].astype("category")
income_mean["income"] = income_mean["income_groups"].cat.codes


print("Income Group Weighted Means - Pearson")
print("Income vs Vehicles:", income_mean["income"].corr(income_mean["weighted_totalvehs"]))
print("Income vs Trip Time:", income_mean["income"].corr(income_mean["weighted_triptime"]))
print("Income vs Trip Distance:", income_mean["income"].corr(income_mean["weighted_cumdist"]))
print("Vehicle vs Trip Time:",income_mean["weighted_totalvehs"].corr(income_mean["weighted_triptime"]))
print("Vehicle vs Trip Distance:",income_mean["weighted_totalvehs"].corr(income_mean["weighted_cumdist"]))

#for nmi, we use binned raw data instead
# bin number can be adjusted, quantile cut so data is more spread out
bins = 10

merged_data["binned_totalvehs"] = pd.qcut(merged_data["totalvehs"], q=bins, duplicates="drop")
merged_data["binned_triptime"] = pd.qcut(merged_data["triptime"], q=bins, duplicates="drop")
merged_data["binned_cumdist"] = pd.qcut(merged_data["cumdist"], q=bins, duplicates="drop")
merged_data["binned_trippurp"] = merged_data["trippurp"].cat.codes
merged_data["binned_mainmode"] = merged_data["mainmode"].cat.codes


# we use average_method = min here because in NMI, we want to have our demoninator
# to be min(H(X), H(Y))
print("Income Group Raw Data - NMI")
print("Income vs Vehicles:", normalized_mutual_info_score(merged_data["weekly_hhinc_group"], merged_data["binned_totalvehs"], average_method='min'))
print("Income vs Trip Time:", normalized_mutual_info_score(merged_data["weekly_hhinc_group"], merged_data["binned_triptime"], average_method='min'))
print("Income vs Trip Distance:", normalized_mutual_info_score(merged_data["weekly_hhinc_group"], merged_data["binned_cumdist"], average_method='min'))
print("Income vs Trip Purpose:", normalized_mutual_info_score(merged_data["weekly_hhinc_group"], merged_data["binned_trippurp"], average_method='min'))
print("Income vs Main Mode:", normalized_mutual_info_score(merged_data["weekly_hhinc_group"], merged_data["binned_mainmode"], average_method='min'))

print("LGA Group Raw Data - NMI")
print("LGA vs Vehicles:", normalized_mutual_info_score(merged_data["homelga"], merged_data["binned_totalvehs"], average_method='min'))
print("LGA vs Trip Time:", normalized_mutual_info_score(merged_data["homelga"], merged_data["binned_triptime"], average_method='min'))
print("LGA vs Trip Distance:", normalized_mutual_info_score(merged_data["homelga"], merged_data["binned_cumdist"], average_method='min'))
print("LGA vs Trip Purpose:", normalized_mutual_info_score(merged_data["homelga"], merged_data["binned_trippurp"], average_method='min'))
print("LGA vs Main Mode:", normalized_mutual_info_score(merged_data["homelga"], merged_data["binned_mainmode"], average_method='min'))

print("Vehicles vs Trip Time:", normalized_mutual_info_score(merged_data["binned_totalvehs"], merged_data["binned_triptime"], average_method='min'))
print("Vehicles vs Trip Distance:", normalized_mutual_info_score(merged_data["binned_totalvehs"], merged_data["binned_cumdist"], average_method='min'))
print("Vehicles vs Trip Purpose:", normalized_mutual_info_score(merged_data["binned_totalvehs"], merged_data["binned_trippurp"], average_method='min'))
print("Income vs LGA:", normalized_mutual_info_score(merged_data["weekly_hhinc_group"], merged_data["homelga"], average_method='min'))


#scatterplots for pearson correlations
pearson_pairs_income = [
    ("income", "weighted_totalvehs", "Income vs Vehicles"),
    ("income", "weighted_triptime", "Income vs Trip Time"),
    ("income", "weighted_cumdist", "Income vs Trip Distance"),
    ("weighted_totalvehs", "weighted_triptime", "Vehicles vs Trip Time"),
    ("weighted_totalvehs", "weighted_cumdist", "Vehicles vs Cumulative Distance"),
    ]

for x, y, title in pearson_pairs_income:
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=x, y=y, data=income_mean, s=100)
    plt.title(f"{title}")
    plt.xlabel(x)
    plt.ylabel(y)
    plt.show()
    
nmi_cols = ["binned_totalvehs", "binned_triptime", "binned_cumdist"]

#normalized heatmap for nmi correlations
heatmap_inc_vehs = pd.crosstab(merged_data["weekly_hhinc_group"], merged_data["binned_totalvehs"], normalize="index")
heatmap_inc_dur = pd.crosstab(merged_data["weekly_hhinc_group"], merged_data["binned_triptime"], normalize="index")
heatmap_inc_dist = pd.crosstab(merged_data["weekly_hhinc_group"], merged_data["binned_cumdist"], normalize="index")
heatmap_inc_purp = pd.crosstab(merged_data["weekly_hhinc_group"], merged_data["trippurp"], normalize="index")
heatmap_inc_mode = pd.crosstab(merged_data["weekly_hhinc_group"], merged_data["mainmode"], normalize="index")

heatmap_lga_vehs = pd.crosstab(merged_data["homelga"], merged_data["binned_totalvehs"], normalize="index")
heatmap_lga_dur = pd.crosstab(merged_data["homelga"], merged_data["binned_triptime"], normalize="index")
heatmap_lga_dist = pd.crosstab(merged_data["homelga"], merged_data["binned_cumdist"], normalize="index")
heatmap_lga_purp = pd.crosstab(merged_data["homelga"], merged_data["trippurp"],normalize="index")
heatmap_lga_mode = pd.crosstab(merged_data["homelga"], merged_data["mainmode"],normalize="index")

plt.figure(figsize=(12,8))
heatmap_inc_vehs = heatmap_inc_vehs.iloc[::-1]
sns.heatmap(heatmap_inc_vehs, annot=True, fmt=".2f")
plt.title("Income Group vs Vehicles")
plt.xlabel("Vehicles")
plt.ylabel("Income Group")
plt.show()

plt.figure(figsize=(12,8))
heatmap_inc_dur = heatmap_inc_dur.iloc[::-1]
sns.heatmap(heatmap_inc_dur, annot=True, fmt=".2f")
plt.title("Income Group vs Trip Time")
plt.xlabel("Trip Time")
plt.ylabel("Income Group")
plt.show()

plt.figure(figsize=(12,8))
heatmap_inc_dist = heatmap_inc_dist.iloc[::-1]
sns.heatmap(heatmap_inc_dist, annot=True, fmt=".2f")
plt.title("Income Group vs Trip Distance")
plt.xlabel("Trip Distance")
plt.ylabel("Income Group")
plt.show()

plt.figure(figsize=(12,8))
heatmap_inc_purp = heatmap_inc_purp.iloc[::-1]
sns.heatmap(heatmap_inc_purp, annot=True, fmt=".2f")
plt.title("Income Group vs Trip Purpose")
plt.xlabel("Trip Purpose")
plt.ylabel("Income Group")
plt.show()

plt.figure(figsize=(12,8))
heatmap_inc_mode = heatmap_inc_mode.iloc[::-1]
sns.heatmap(heatmap_inc_mode, annot=True, fmt=".2f")
plt.title("Income Group vs Main Mode")
plt.xlabel("Main Mode")
plt.ylabel("Income Group")
plt.show()


plt.figure(figsize=(16,8))
sns.heatmap(heatmap_lga_vehs, annot=True, fmt=".2f")
plt.title("LGA Group vs Vehicles")
plt.xlabel("Vehicles")
plt.ylabel("LGA Group")
plt.show()

plt.figure(figsize=(16,8))
sns.heatmap(heatmap_lga_dur, annot=True, fmt=".2f")
plt.title("LGA Group vs Trip Time")
plt.xlabel("Trip Time")
plt.ylabel("LGA Group")
plt.show()

plt.figure(figsize=(16,8))
sns.heatmap(heatmap_lga_dist, annot=True, fmt=".2f")
plt.title("LGA Group vs Trip Distance")
plt.xlabel("Trip Distance")
plt.ylabel("LGA Group")
plt.show()

plt.figure(figsize=(16,8))
sns.heatmap(heatmap_lga_purp, annot=True, fmt=".2f")
plt.title("LGA Group vs Trip Purpose")
plt.xlabel("Trip Purpose")
plt.ylabel("LGA Group")
plt.show()

plt.figure(figsize=(16,8))
sns.heatmap(heatmap_lga_mode, annot=True, fmt=".2f")
plt.title("LGA Group vs Main Mode")
plt.xlabel("Main Mode")
plt.ylabel("LGA Group")
plt.show()
