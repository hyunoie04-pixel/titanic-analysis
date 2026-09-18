import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

df = pd.read_csv("data/train_and_test2.csv")

df = df.rename(columns={"Passengerid": "PassengerId", "2urvived": "Survived"})

zero_cols = [c for c in df.columns if c.startswith("zero")]
df = df.drop(columns=zero_cols)
print(f"[INFO] Dropped {len(zero_cols)} 'zero' columns (constant, no information)\n")

print("=" * 60)
print("1. BASIC INFO")
print("=" * 60)
print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
print("\nColumn types:")
print(df.dtypes)
print("\nFirst 5 rows:")
print(df.head())
print("\nNumeric summary (describe):")
print(df.describe().round(2))

print("\n" + "=" * 60)
print("2. MISSING VALUES (NaN check)")
print("=" * 60)
missing = df.isna().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_table = pd.DataFrame({"missing_count": missing, "missing_pct": missing_pct})
missing_table = missing_table.sort_values("missing_count", ascending=False)
print(missing_table[missing_table["missing_count"] > 0].to_string() if missing_table["missing_count"].sum() > 0 else "No missing values found.")
if missing_table["missing_count"].sum() > 0:
    print("\nRows containing missing values:")
    print(df[df.isna().any(axis=1)])

print("\n" + "=" * 60)
print("3. DUPLICATES")
print("=" * 60)
dup_rows = df.duplicated().sum()
print(f"Duplicate rows (all columns identical): {dup_rows}")
dup_ids = df["PassengerId"].duplicated().sum()
print(f"Duplicate PassengerId values: {dup_ids}")

print("\n" + "=" * 60)
print("4. OUTLIERS (IQR method: outside Q1-1.5*IQR ~ Q3+1.5*IQR)")
print("=" * 60)
numeric_cols = ["Age", "Fare", "sibsp", "Parch"]
outlier_bounds = {}
for col in numeric_cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    mask = (df[col] < lower) | (df[col] > upper)
    n_out = int(mask.sum())
    outlier_bounds[col] = (lower, upper)
    print(f"{col:6s}: Q1={q1:8.2f}, Q3={q3:8.2f}, IQR={iqr:8.2f}, "
          f"bounds=[{lower:8.2f}, {upper:8.2f}], outliers={n_out:4d} ({n_out / len(df) * 100:.1f}%)")
print("\nTop 10 most extreme Fares:")
print(df.nlargest(10, "Fare")[["PassengerId", "Fare", "Pclass", "Survived"]])

print("\n[PLOT] Saving charts to 'plots/' ...")

plt.figure(figsize=(8, 5))
sns.barplot(x=missing_table["missing_count"], y=missing_table.index)
plt.title("Missing Value Count per Column")
plt.xlabel("Missing Count")
plt.tight_layout()
plt.savefig("plots/1_missing_values.png")
plt.close()

plt.figure(figsize=(6, 4))
sns.heatmap(df.isna(), cbar=False, yticklabels=False)
plt.title("Missing Value Heatmap (yellow = missing)")
plt.tight_layout()
plt.savefig("plots/2_missing_heatmap.png")
plt.close()

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, col in zip(axes.flat, numeric_cols):
    sns.boxplot(x=df[col], ax=ax, color="skyblue")
    ax.set_title(f"{col} boxplot (whiskers = outlier bounds)")
plt.tight_layout()
plt.savefig("plots/3_boxplots.png")
plt.close()

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(df["Age"], bins=30, kde=True, ax=axes[0])
axes[0].set_title("Age distribution")
sns.histplot(df["Fare"], bins=50, kde=True, ax=axes[1])
axes[1].set_title("Fare distribution")
plt.tight_layout()
plt.savefig("plots/4_histograms.png")
plt.close()

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.boxplot(data=df, x="Survived", y="Age", ax=axes[0])
sns.boxplot(data=df, x="Survived", y="Fare", ax=axes[1])
plt.tight_layout()
plt.savefig("plots/5_age_fare_by_survived.png")
plt.close()

print("[DONE] Charts saved: plots/1_missing_values.png, 2_missing_heatmap.png, 3_boxplots.png, 4_histograms.png, 5_age_fare_by_survived.png")
