import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import normalized_mutual_info_score


#load and merge all the data
household = pd.read_csv("cleaned_household.csv")
household_weights = pd.read_csv("cleaned_household_weights.csv")
trips = pd.read_csv("cleaned_trips.csv")
trip_weights = pd.read_csv("cleaned_trips_weights.csv")

household = household.merge(household_weights[["hhid", "hhpoststratweight"]], on="hhid", how="left")

merged_data = trips.merge(household, on="hhid", how="left")

merged_data = merged_data.merge(trip_weights[["tripid","trippoststratweight"]], on="tripid", how="left")

merged_data["hhpoststratweight"] = merged_data["hhpoststratweight"].fillna(1)
merged_data["trippoststratweight"] = merged_data["trippoststratweight"].fillna(1)

#categorical data
merged_data["weekly_hhinc_group"] = merged_data["weekly_hhinc_group"].astype("category")
merged_data["homelga"] = merged_data["homelga"].astype("category")
merged_data["trippurp"] = merged_data["trippurp"].astype("category")
merged_data["mainmode"] = merged_data["mainmode"].astype("category")


#numerical data
merged_data["totalvehs"] = pd.to_numeric(merged_data["totalvehs"], errors="coerce")
merged_data["duration"] = pd.to_numeric(merged_data["duration"], errors="coerce")
merged_data["cumdist"] = pd.to_numeric(merged_data["cumdist"], errors="coerce")

#remove rows with missing values
merged_data = merged_data.dropna(subset=["weekly_hhinc_group","homelga","mainmode",
                                         "trippurp", "totalvehs", "duration", "cumdist"])


#compute the weighted means (income)

income_mean = []
for group in merged_data["weekly_hhinc_group"].unique():
    column = merged_data[merged_data["weekly_hhinc_group"] == group]
    weighted_vehs = (column["totalvehs"] * column["hhpoststratweight"]).sum()/ column["hhpoststratweight"].sum()
    weighted_dur = (column["duration"] * column["trippoststratweight"]).sum()/ column["trippoststratweight"].sum()
    weighted_dist = (column["cumdist"] * column["trippoststratweight"]).sum()/ column["trippoststratweight"].sum()
    income_mean.append({
        "income_groups": group,
        "weighted_totalvehs": weighted_vehs,
        "weighted_duration": weighted_dur,
        "weighted_cumdist": weighted_dist
        })

income_mean = pd.DataFrame(income_mean)
income_mean["income_groups"] = income_mean["income_groups"].astype("category")
income_mean["income"] = income_mean["income_groups"].cat.codes

#compute the weighted means (lga)

lga_mean = []
for group in merged_data["homelga"].unique():
    column = merged_data[merged_data["homelga"] == group]
    weighted_vehs = (column["totalvehs"] * column["hhpoststratweight"]).sum()/ column["hhpoststratweight"].sum()
    weighted_dur = (column["duration"] * column["trippoststratweight"]).sum()/ column["trippoststratweight"].sum()
    weighted_dist = (column["cumdist"] * column["trippoststratweight"]).sum()/ column["trippoststratweight"].sum()
    lga_mean.append({
        "lga_groups": group,
        "weighted_totalvehs": weighted_vehs,
        "weighted_duration": weighted_dur,
        "weighted_cumdist": weighted_dist
        })

lga_mean = pd.DataFrame(lga_mean)
lga_mean["lga_groups"] = lga_mean["lga_groups"].astype("category")
lga_mean["lga"] = lga_mean["lga_groups"].cat.codes


print("Income Group Weighted Means - Pearson")
print("Income vs Vehicles:", income_mean["income"].corr(income_mean["weighted_totalvehs"]))
print("Income vs Duration:", income_mean["income"].corr(income_mean["weighted_duration"]))
print("Income vs Trip Distance:", income_mean["income"].corr(income_mean["weighted_cumdist"]))

print("LGA Group Weighted Means - Pearson")
print("LGA vs Vehicles:", lga_mean["lga"].corr(lga_mean["weighted_totalvehs"]))
print("LGA vs Duration:", lga_mean["lga"].corr(lga_mean["weighted_duration"]))
print("LGA vs Trip Distance:", lga_mean["lga"].corr(lga_mean["weighted_cumdist"]))

#for nmi, we use binned raw data instead
# bin number can be adjusted, just using 10 because max vehicles is 10
bins = 10

merged_data["binned_totalvehs"] = pd.cut(merged_data["totalvehs"], bins=bins)
merged_data["binned_duration"] = pd.cut(merged_data["duration"], bins=bins)
merged_data["binned_cumdist"] = pd.cut(merged_data["cumdist"], bins=bins)
merged_data["binned_trippurp"] = merged_data["trippurp"].cat.codes
merged_data["binned_mainmode"] = merged_data["mainmode"].cat.codes


print("Income Group Raw Data - NMI")
print("Income vs Vehicles:", normalized_mutual_info_score(merged_data["weekly_hhinc_group"], merged_data["binned_totalvehs"]))
print("Income vs Duration:", normalized_mutual_info_score(merged_data["weekly_hhinc_group"], merged_data["binned_duration"]))
print("Income vs Trip Distance:", normalized_mutual_info_score(merged_data["weekly_hhinc_group"], merged_data["binned_cumdist"]))
print("Income vs Trip Purpose:", normalized_mutual_info_score(merged_data["weekly_hhinc_group"], merged_data["binned_trippurp"]))
print("Income vs Main Mode:", normalized_mutual_info_score(merged_data["weekly_hhinc_group"], merged_data["binned_mainmode"]))

print("LGA Group Raw Data - NMI")
print("LGA vs Vehicles:", normalized_mutual_info_score(merged_data["homelga"], merged_data["binned_totalvehs"]))
print("LGA vs Duration:", normalized_mutual_info_score(merged_data["homelga"], merged_data["binned_duration"]))
print("LGA vs Trip Distance:", normalized_mutual_info_score(merged_data["homelga"], merged_data["binned_cumdist"]))
print("LGA vs Trip Purpose:", normalized_mutual_info_score(merged_data["homelga"], merged_data["binned_trippurp"]))
print("LGA vs Main Mode:", normalized_mutual_info_score(merged_data["homelga"], merged_data["binned_mainmode"]))

#scatterplots for pearson correlations
pearson_pairs_income = [
    ("income", "weighted_totalvehs", "Income vs Vehicles"),
    ("income", "weighted_duration", "Income vs Duration"),
    ("income", "weighted_cumdist", "Income vs Trip Distance"),
    ]
pearson_pairs_lga = [
    ("lga", "weighted_totalvehs", "LGA vs Vehicles"),
    ("lga", "weighted_duration", "LGA vs Duration"),
    ("lga", "weighted_cumdist", "LGA vs Trip Distance"),
    ]

for x, y, title in pearson_pairs_income:
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=x, y=y, data=income_mean, s=100)
    plt.title(f"{title}")
    plt.xlabel(x)
    plt.ylabel(y)
    plt.show()

for x, y, title in pearson_pairs_lga:
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=x, y=y, data=lga_mean, s=100)
    plt.title(f"{title}")
    plt.xlabel(x)
    plt.ylabel(y)
    plt.show()
    
nmi_cols = ["binned_totalvehs", "binned_duration", "binned_cumdist"]

#heatmap for nmi correlations
heatmap_inc_vehs = pd.crosstab(merged_data["weekly_hhinc_group"], merged_data["binned_totalvehs"])
heatmap_inc_dur = pd.crosstab(merged_data["weekly_hhinc_group"], merged_data["binned_duration"])
heatmap_inc_dist = pd.crosstab(merged_data["weekly_hhinc_group"], merged_data["binned_cumdist"])
heatmap_inc_purp = pd.crosstab(merged_data["weekly_hhinc_group"], merged_data["trippurp"])
heatmap_inc_mode = pd.crosstab(merged_data["weekly_hhinc_group"], merged_data["mainmode"])

heatmap_lga_vehs = pd.crosstab(merged_data["homelga"], merged_data["binned_totalvehs"])
heatmap_lga_dur = pd.crosstab(merged_data["homelga"], merged_data["binned_duration"])
heatmap_lga_dist = pd.crosstab(merged_data["homelga"], merged_data["binned_cumdist"])
heatmap_lga_purp = pd.crosstab(merged_data["homelga"], merged_data["trippurp"])
heatmap_lga_mode = pd.crosstab(merged_data["homelga"], merged_data["mainmode"])

plt.figure(figsize=(12,8))
sns.heatmap(heatmap_inc_vehs, annot=True, fmt="d")
plt.title("Income Group vs Vehicles")
plt.xlabel("Vehicles")
plt.ylabel("Income Group")
plt.show()

plt.figure(figsize=(12,8))
sns.heatmap(heatmap_inc_dur, annot=True, fmt="d")
plt.title("Income Group vs Duration")
plt.xlabel("Duration")
plt.ylabel("Income Group")
plt.show()

plt.figure(figsize=(12,8))
sns.heatmap(heatmap_inc_dist, annot=True, fmt="d")
plt.title("Income Group vs Trip Distance")
plt.xlabel("Trip Distance")
plt.ylabel("Income Group")
plt.show()

plt.figure(figsize=(12,8))
sns.heatmap(heatmap_inc_purp, annot=True, fmt="d")
plt.title("Income Group vs Trip Purpose")
plt.xlabel("Trip Purpose")
plt.ylabel("Income Group")
plt.show()

plt.figure(figsize=(12,8))
sns.heatmap(heatmap_inc_mode, annot=True, fmt="d")
plt.title("Income Group vs Main Mode")
plt.xlabel("Main Mode")
plt.ylabel("Income Group")
plt.show()


plt.figure(figsize=(16,8))
sns.heatmap(heatmap_lga_vehs, annot=True, fmt="d")
plt.title("LGA Group vs Vehicles")
plt.xlabel("Vehicles")
plt.ylabel("LGA Group")
plt.show()

plt.figure(figsize=(16,8))
sns.heatmap(heatmap_lga_dur, annot=True, fmt="d")
plt.title("LGA Group vs Duration")
plt.xlabel("Duration")
plt.ylabel("LGA Group")
plt.show()

plt.figure(figsize=(16,8))
sns.heatmap(heatmap_lga_dist, annot=True, fmt="d")
plt.title("LGA Group vs Trip Distance")
plt.xlabel("Trip Distance")
plt.ylabel("LGA Group")
plt.show()

plt.figure(figsize=(16,8))
sns.heatmap(heatmap_lga_purp, annot=True, fmt="d")
plt.title("LGA Group vs Trip Purpose")
plt.xlabel("Trip Purpose")
plt.ylabel("LGA Group")
plt.show()

plt.figure(figsize=(16,8))
sns.heatmap(heatmap_lga_mode, annot=True, fmt="d")
plt.title("LGA Group vs Main Mode")
plt.xlabel("Main Mode")
plt.ylabel("LGA Group")
plt.show()

