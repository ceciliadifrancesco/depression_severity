#!/usr/bin/env python
# coding: utf-8

# In[161]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

pd.set_option('display.max_columns', None)

# loading the data file
df = pd.read_parquet(r"C:\Users\cecil\Desktop\Digital health\project\depression_severity\data\raw\anon_processed_df_parquet")
df.head()


# In[162]:


df = df.copy()

df.index = df.index.astype(str)
# extracting participant_id and month from the index
df['participant_id'] = df.index.str.split("_").str[0].astype(int)
df['month'] = df.index.str.split("_").str[1].astype(int)

df.head()


# #### Basic data exploration

# In[163]:


print("\nDataFrame Info:")
df.info()

print("\nSummary statistics (numeric columns):")
display(df.describe())


# In[164]:


num_rows = df.shape[0]
print(f"Number of rows in the dataset: {num_rows}")# count unique participants
num_participants = df['participant_id'].nunique()
print(f"Number of unique participants: {num_participants}")
# count how many months of data
num_months = df['month'].nunique()
print(f"Number of months of data: {num_months}")
# count how many years of data
num_years = num_months / 12
print(f"Number of years of data: {num_years}")


# In[165]:


num_duplicates = df.duplicated().sum()
print(f"\nNumber of duplicate rows: {num_duplicates}")


participant_months = df.groupby("participant_id")["month"].nunique().describe()
print("\nDistribution of number of months available per participant:")
display(participant_months)


# In[166]:


df["sex"].value_counts()


# ### Missing Data Analysis

# In[ ]:


missing_df = pd.DataFrame({
    "missing_count": df.isna().sum(),
    "missing_percent": (df.isna().sum() / len(df)) * 100
})


missing_df = missing_df.sort_values("missing_percent", ascending=False)

missing_df["missing_percent"] = missing_df["missing_percent"].round(2)

print(missing_df)


# ### Removing unnecessary variables

# In[168]:


# Patterns of columns to remove
patterns_to_drop = [
    "score", "intercept", "coeff", "iqr", "range"
]

protected = ["phq9_score_start", "phq9_score_end"]

cols_to_drop = [
    col for col in df.columns
    if any(p in col.lower() for p in patterns_to_drop)
    and col not in protected
]
len(cols_to_drop), cols_to_drop[:]  # checking what will be removed


# In[169]:


df_clean = df.drop(columns=cols_to_drop)
df_clean.shape


# In[170]:


df_clean.info()
df = df_clean.copy()


# #### Checking missing values after removing variables

# In[ ]:


# creating feature groups based on column name patterns to create some broeader categories
sleep_features = [c for c in df.columns if "sleep" in c.lower()]
step_features = [c for c in df.columns if "step" in c.lower()]
phq_features = [c for c in df.columns if "phq" in c.lower()]
lifestyle_features = [c for c in df.columns if "life" in c.lower()]
medication_features = [c for c in df.columns if "med" in c.lower()]
demographic_features = ["sex","birthyear","height","weight","bmi","pregnant","insurance"]

feature_groups = {
    "sleep": sleep_features,
    "steps": step_features,
    "phq": phq_features,
    "lifestyle": lifestyle_features,
    "medication": medication_features,
    "demographics": demographic_features
}

group_missing = {
    group: df[cols].isna().mean().mean() * 100 
    for group, cols in feature_groups.items() if cols
}

group_missing


# In[172]:


fig, ax = plt.subplots(figsize=(10,6))
group_missing_no_phq = {k: v for k, v in group_missing.items() if k != "phq"}

ax.bar(group_missing_no_phq.keys(),
       group_missing_no_phq.values(),
       color="#6a5acd", edgecolor="black", linewidth=1)


ax.spines['bottom'].set_visible(False)

ax.set_ylabel("Average % Missing", fontsize=12)
ax.set_title("Missingness by Feature Group", fontsize=14)
ax.set_xticklabels(group_missing_no_phq.keys(), rotation=20, fontsize=11)


for i, (group, value) in enumerate(group_missing_no_phq.items()):
    ax.text(i, value + 0.05, f"{value:.2f}%", ha="center", fontsize=10)

plt.tight_layout()
plt.show()


# #### Data Exploration

# In[173]:


phq_counts = df.groupby("participant_id")[["phq9_score_start","phq9_score_end"]].count()
phq_counts["total_phq"] = phq_counts.sum(axis=1)

phq_counts["total_phq"].describe()


# In[174]:


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))
sns.histplot(df["phq9_score_start"], bins=20, kde=True, color="orange")
plt.title("Distribution of PHQ-9 Start Scores")
plt.xlabel("PHQ-9 Score")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(10,5))
sns.histplot(df["phq9_score_end"], bins=20, kde=True, color="green")
plt.title("Distribution of PHQ-9 End Scores")
plt.xlabel("PHQ-9 Score")
plt.ylabel("Count")
plt.show()


# In[175]:


df["phq9_change"] = df["phq9_score_end"] - df["phq9_score_start"]

sns.histplot(df["phq9_change"], bins=20, kde=True, color="purple")
plt.title("Distribution of PHQ-9 Change over 3-Month Interval")
plt.xlabel("PHQ-9 Change (End - Start)")
plt.ylabel("Count")
plt.show()


# In[176]:


numeric_df = df.select_dtypes(include=['float64', 'int64'])


corrs = numeric_df.corr()["phq9_change"].sort_values()


corrs.head(15), corrs.tail(15)


# In[177]:


top_features = corrs.tail(10).index.tolist()

plt.figure(figsize=(8,6))
sns.heatmap(numeric_df[top_features].corr().round(2), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap - Highest Correlation Behaviour Features and PHQ-9 Change")
plt.show()


# In[178]:


df["phq_group"] = pd.cut(
    df["phq9_change"],
    bins=[-100, -2, 2, 100],  # wide bounds
    labels=["Improved", "Stable", "Worsened"]
)

df["phq_group"].value_counts()


# #### Visualizing behavioural trends across PHQ-9 severity categories

# In[179]:


df.describe(include='all')


# In[180]:


# Sleep vs PHQ-9 Change

# Selecting the most relevant sleep variables
sleep_vars = [
    "sleep_asleep_weekday_mean",
    "sleep_asleep_weekend_mean",
    "sleep_ratio_asleep_in_bed_mean_recent",
    "sleep_main_start_hour_adj_median"
]

plt.figure(figsize=(14,10))

for i, var in enumerate(sleep_vars, 1):
    plt.subplot(2,2,i)
    sns.boxplot(data=df, x="phq_group", y=var, hue = "phq_group", palette="viridis", legend=False)
    plt.title(f"{var.replace('_',' ').title()} Across PHQ-9 Change Groups")
    plt.xlabel("")
    plt.ylabel(var)

plt.tight_layout()
plt.show()

df.groupby("phq_group", observed=False)[sleep_vars].mean()


# In[181]:


# Stress vs PHQ-9 Change
stress_props = df.groupby("phq_group", observed=False)["life_stress"].mean().reset_index()
stress_props["life_stress"] *= 100  

plt.figure(figsize=(10,5))
sns.barplot(data=stress_props, x="phq_group", y="life_stress", color="purple")
plt.title("Percentage of Participants Reporting High Stress by PHQ-9 Change Group")
plt.ylabel("% Reporting Stress")
plt.xlabel("PHQ-9 Change Group")
plt.ylim(0, 100)
plt.show()


# In[182]:


# Sex vs PHQ-9 Change
plt.figure(figsize=(6,4))
sns.countplot(data=df, x="phq_group", hue="sex")
plt.title("Sex Distribution Across PHQ-9 Change Groups")
plt.xlabel("PHQ-9 Change Group")
plt.ylabel("Count")
plt.legend(title="Sex (0 = female, 1 = male. 2 = other)")
plt.show()


# In[183]:


# Activity vs PHQ-9 Change
# Choosing relevant activity variables
activity_vars = [
    "steps_awake_mean",
    "steps_mvpa_sum_recent",
    "steps_lpa_sum_recent",
    "steps_rolling_6_median_recent"
]

plt.figure(figsize=(14,10))

for i, var in enumerate(activity_vars, 1):
    plt.subplot(2,2,i)
    sns.boxplot(data=df, x="phq_group", y=var, hue="phq_group",
    palette="magma",
    legend=False)
    plt.title(f"{var.replace('_',' ').title()} Across PHQ-9 Change Groups")
    plt.xlabel("")
    plt.ylabel(var)

plt.tight_layout()
plt.show()

df.groupby("phq_group", observed=True)[activity_vars].mean()


# In[184]:


# Trauma vs PHQ-9 Change
plt.figure(figsize=(6,4))
sns.barplot(data=df, x="phq_group", y="trauma", hue="phq_group",
    palette="Purples",
    legend=False)
plt.title("Trauma Prevalence Across PHQ-9 Change Groups")
plt.ylabel("Proportion with Trauma History")
plt.xlabel("")
plt.show()

df.groupby("phq_group", observed=False)["trauma"].mean()


# In[186]:


# creating has migraine column
df["has_migraine"] = (df["comorbid_migraines"] == 1) | (df["num_migraine_days"] > 0)


# In[187]:


# checking migraine presence in the dataset
df["has_migraine"].value_counts()


# In[189]:


# Age vs PHQ-9 Change

# calculating age from birthyear
current_year = 2019 # considering the midpoint of data collection (2018-2020)
df["age"] = current_year - df["birthyear"]
df[["age", "phq9_change"]].corr().round(3)



# In[190]:


df["age"].describe()


# In[191]:


# distribution of age in the dataset
sns.boxplot(data=df, x="phq_group", y="age", color = "lightpink")
plt.title("Age Distribution Across PHQ-9 Change Groups")
plt.show()


# In[192]:


sns.lmplot(data=df, x="age", y="phq9_change", scatter_kws={'alpha':0.3})
plt.title("PHQ-9 Change vs Age")
plt.show()


# In[193]:


sns.boxplot(data=df, x="phq_group", y="age", color = "lightblue")
plt.title("Age Distribution Across PHQ-9 Change Groups")
plt.show()


# #### Filtering rows where we have labels

# In[194]:


# Filter rows where PHQ-9 score is defined
df = df[df["phq9_change"].notna()].copy()
df.shape


# In[195]:


# check missing values
for col in df.columns:
    print(f"{col}: {df[col].isna().sum()} missing values")


# In[ ]:


# imputation
continuous_vars = [
    "steps_awake_mean", "sleep_asleep_weekday_mean", "sleep_asleep_weekend_mean",
    "sleep_in_bed_weekday_mean", "sleep_in_bed_weekend_mean",
    "sleep_ratio_asleep_in_bed_weekday_mean", "sleep_ratio_asleep_in_bed_weekend_mean",
    "sleep_main_start_hour_adj_median", "sleep_asleep_mean_recent",
    "sleep_in_bed_mean_recent", "sleep_ratio_asleep_in_bed_mean_recent",
    "steps_rolling_6_median_recent", "steps_rolling_6_max_recent",
    "educ", "height", "weight", "bmi", "pregnant", "birth", "insurance",
    "money", "money_assistance"
]

median_imputer = SimpleImputer(strategy="median")
df[continuous_vars] = median_imputer.fit_transform(df[continuous_vars])


categorical_vars = ["sex"]
freq_imputer = SimpleImputer(strategy="most_frequent")
df[categorical_vars] = freq_imputer.fit_transform(df[categorical_vars])


# In[197]:


# check missing values again
for col in df.columns:
    print(f"{col}: {df[col].isna().sum()} missing values")


# ### Modelling

# In[199]:


vars_to_drop = [
    "phq9_score_end", "phq9_cat_end",
    "participant_id", "month",
    "pregnant", "birth", "money", "money_assistance",
    "insurance", "household",
    "comorbid_cancer", "comorbid_diabetes_typ1", "comorbid_diabetes_typ2",
    "comorbid_gout", "comorbid_migraines", "comorbid_ms",
    "comorbid_osteoporosis", "comorbid_neuropathic", "comorbid_arthritis",
    "med_start", "med_stop", "med_dose",
    "nonmed_start", "nonmed_stop", "med_nonmed_dnu",
    "meds_migraine", "num_migraine_days"
]

df = df.drop(columns=[col for col in vars_to_drop if col in df.columns])


# In[200]:


# Variables recommended for clustering
cluster_vars = [
    # Activity
    "steps_awake_mean", "steps_mvpa_sum_recent", "steps_lpa_sum_recent",
    "steps_rolling_6_median_recent", "steps_rolling_6_max_recent",

    # Sleep
    "sleep_asleep_weekday_mean", "sleep_asleep_weekend_mean",
    "sleep_in_bed_weekday_mean", "sleep_in_bed_weekend_mean",
    "sleep_ratio_asleep_in_bed_weekday_mean",
    "sleep_ratio_asleep_in_bed_weekend_mean",
    "sleep_main_start_hour_adj_median",
    "sleep_asleep_mean_recent", "sleep_in_bed_mean_recent",
    "sleep_ratio_asleep_in_bed_mean_recent",

    "sleep__hypersomnia_count_", "sleep__hyposomnia_count_",
    "steps__active_day_count_", "steps__sedentary_day_count_",

    # Lifestyle
    "life_meditation", "life_stress", "life_activity_eating",
    "life_red_stop_alcoh",

    # Demographics
    "age", "sex", "bmi"
]


# In[201]:


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Keep only rows with no missing values in these columns
df_cluster = df[cluster_vars].dropna().copy()

# Standardize all features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_cluster)


# In[212]:


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np

sil_scores = []
K_values = range(2, 10)

for k in K_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels)
    sil_scores.append(sil)
    print(f"K={k}, Silhouette Score={sil:.4f}")

# Plot
plt.figure(figsize=(7, 5))
plt.plot(K_values, sil_scores, marker="o")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score for Different K")
plt.grid(True)
plt.show()


# In[213]:


k = 4
kmeans = KMeans(n_clusters=k, random_state=42)
df["cluster_kmeans"] = kmeans.fit_predict(X_scaled)


# In[214]:


# Reduce to 2 principal components for plotting
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

df["pca1"] = X_pca[:, 0]
df["pca2"] = X_pca[:, 1]

plt.figure(figsize=(7, 6))
plt.scatter(df["pca1"], df["pca2"], c=df["cluster_kmeans"], alpha=0.6)
plt.title("K-Means Clusters Visualized in PCA Space")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.colorbar(label="Cluster")
plt.show()


# In[217]:


cluster_centroids = pd.DataFrame(
    kmeans.cluster_centers_,
    columns=cluster_vars
)

cluster_centroids


# In[218]:


import pandas as pd
import numpy as np

# Attach cluster labels to df_cluster
df_cluster["cluster"] = kmeans.labels_

# 1. Compute mean of each feature by cluster
cluster_profiles = df_cluster.groupby("cluster")[cluster_vars].mean()

# 2. Standardized profiles (z-scores relative to whole population)
feature_means = df_cluster[cluster_vars].mean()
feature_stds = df_cluster[cluster_vars].std()

cluster_profiles_z = (cluster_profiles - feature_means) / feature_stds

cluster_profiles, cluster_profiles_z


# In[219]:


cluster_top_features = {}

for c in cluster_profiles_z.index:
    sorted_feats = cluster_profiles_z.loc[c].abs().sort_values(ascending=False)
    cluster_top_features[c] = sorted_feats.head(5)

pd.DataFrame(cluster_top_features)


# In[220]:


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12,6))
sns.heatmap(cluster_profiles_z, cmap="coolwarm", center=0, annot=False)
plt.title("Cluster Profiles (Z-Scores)")
plt.show()


# In[221]:


# Map clusters back into the full dataset (if desired)
df["cluster"] = df_cluster["cluster"]

# Count distribution
display(df["cluster"].value_counts())

# Compare PHQ-9 change across clusters
df.groupby("cluster")["phq9_change"].mean()

