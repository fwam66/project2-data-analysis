import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import normalized_mutual_info_score

merged_data = pd.read_csv("merged_data.csv")

merged_data["duration"] = pd.to_numeric(merged_data["duration"],errors="coerce")
merged_data["cumdist"] = pd.to_numeric(merged_data["cumdist"],errors="coerce")
merged_data["totalvehs"] = pd.to_numeric(merged_data["totalvehs"],errors="coerce")

merged_data = merged_data.dropna(subset=["duration","cumdist","weekly_hhinc_group",
                                         "mainmode","totalvehs","dayType","homeregion_ASGS"])



#income vs trip distance/time/mode
x_income = merged_data["weekly_hhinc_group"].astype("category").cat.codes
y_dist = merged_data["cumdist"].astype("category").cat.codes
y_duration = merged_data["duration"].astype("category").cat.codes
y_mode = merged_data["mainmode"].astype("category").cat.codes


#nmi for income vs trip distance/time/mode
nmi_inc_dist = normalized_mutual_info_score(x_income, y_dist, average_method="min")
nmi_inc_duration = normalized_mutual_info_score(x_income, y_duration, average_method="min")
nmi_inc_mode = normalized_mutual_info_score(x_income, y_mode, average_method="min")

print("Normalized Mutual Information (Income vs Trip Distance):", nmi_inc_dist)
print("Normalized Mutual Information (Income vs Trip Duration):", nmi_inc_duration)
print("Normalized Mutual Information (Income vs Mode):", nmi_inc_mode)

#nmi for vehicles owned vs car/non-car usage
x_vehs = merged_data["totalvehs"].astype("category").cat.codes

nmi_vehs_mode = normalized_mutual_info_score(x_vehs, y_mode, average_method="min")

print("Normalized Mutual Information (Vehicles vs Mode):", nmi_vehs_mode)

#nmi for region vs weekday/weekend
x_region = merged_data["homeregion_ASGS"].astype("category").cat.codes
y_daytype = merged_data["dayType"].astype("category").cat.codes

nmi_region_daytype = normalized_mutual_info_score(x_region, y_daytype, average_method="min")

print("Normalized Mutual Information (Regions vs Day Type):", nmi_region_daytype)


#bins number can be adjusted, this is only for plotting the data
merged_data["inc_bin"] = pd.cut(merged_data["weekly_hhinc_group"], bins=10)
merged_data["dist_bin"] = pd.cut(merged_data["cumdist"], bins=10)
merged_data["dur_bin"] = pd.cut(merged_data["duration"], bins=10)
merged_data["vehs_bin"] = pd.cut(merged_data["totalvehs"], bins=10)


#heatmap of nmi_inc_dist
plt.figure(figsize=(10,6))
heatmap = pd.crosstab(merged_data["inc_bin"], merged_data["dist_bin"])
sns.heatmap(heatmap, annot=True, fmt="d")
plt.title("Heatmap of HH Income vs Cummulative Distance")
plt.xlabel("Cummulative Distance")
plt.ylabel("HH Income")
plt.show()

#heatmap of nmi_inc_duration
plt.figure(figsize=(10,6))
heatmap = pd.crosstab(merged_data["inc_bin"], merged_data["dur_bin"])
sns.heatmap(heatmap, annot=True, fmt="d")
plt.title("Heatmap of HH Income vs Duration")
plt.xlabel("Trip Duration")
plt.ylabel("HH Income")
plt.show()

#heatmap of nmi_inc_mode
plt.figure(figsize=(10,6))
heatmap = pd.crosstab(merged_data["inc_bin"], merged_data["mainmode"])
sns.heatmap(heatmap, annot=True, fmt="d")
plt.title("Heatmap of HH Income vs Mode")
plt.xlabel("Mode")
plt.ylabel("HH Income")
plt.show()

#heatmap of nmi_vehs_mode
plt.figure(figsize=(10,6))
heatmap = pd.crosstab(merged_data["vehs_bin"], merged_data["mainmode"])
sns.heatmap(heatmap, annot=True, fmt="d")
plt.title("Heatmap of Total Vehicles vs Mode")
plt.xlabel("Mode")
plt.ylabel("Total Vehicles")
plt.show()

#heatmap of nmi_region_daytype
plt.figure(figsize=(10,6))
heatmap = pd.crosstab(merged_data["homeregion_ASGS"], merged_data["dayType"])
sns.heatmap(heatmap, annot=True, fmt="d")
plt.title("Heatmap of Region vs Daytype")
plt.xlabel("Region")
plt.ylabel("Daytype")
plt.show()
