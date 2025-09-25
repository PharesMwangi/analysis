# Task 1: Load and Explore the Dataset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
import matplotlib
from matplotlib import rcsetup

# Load dataset
try:
    iris_data = load_iris(as_frame=True)
    df = iris_data.frame
    print("✅ Dataset loaded successfully!\n")
except FileNotFoundError:
    print("❌ File not found. Please check the dataset path.")
except Exception as e:
    print("❌ Error loading dataset:", e)

# Display first few rows
print("🔹 First 5 rows of the dataset:")
# In notebooks `display` is available; in scripts it's not. Try to use IPython's display,
# otherwise fall back to printing the DataFrame.
if 'df' in locals():
    try:
        from IPython.display import display as _display
        _display(df.head())
    except Exception:
        print(df.head())
else:
    print("Dataset not available to display.")

# Explore dataset structure
print("\n🔹 Dataset info:")
print(df.info())

print("\n🔹 Missing values per column:")
print(df.isnull().sum())

# Clean dataset to check if there were missing values
df = df.dropna()

# Task 2: Basic Data Analysis
print("\n🔹 Descriptive statistics:")
print(df.describe())

# Grouping by species
print("\n🔹 Average values per species:")
species_group = df.groupby("target").mean()
print(species_group)

# Replace numeric target with species names for readability
df["species"] = df["target"].map({i: name for i, name in enumerate(iris_data.target_names)})

print("\n🔹 Average petal length by species:")
print(df.groupby("species")["petal length (cm)"].mean())

# Task 3: Data Visualization
sns.set_style("whitegrid")

# 1. Line chart - simulate as if petal length varies across samples
plt.figure(figsize=(8,5))
plt.plot(df.index, df["petal length (cm)"], label="Petal Length")
plt.title("Line Chart: Petal Length Across Samples")
plt.xlabel("Sample Index")
plt.ylabel("Petal Length (cm)")
plt.legend()

backend = matplotlib.get_backend()
interactive_backends = [b.lower() for b in rcsetup.interactive_bk]
if backend.lower() in interactive_backends:
    plt.show()
else:
    out_file = "petal_length_line.png"
    plt.savefig(out_file, bbox_inches="tight")
    print(f"🔹 Non-interactive backend ({backend}) detected — saved plot to '{out_file}'")

# 2. Bar chart - average petal length per species
plt.figure(figsize=(6,5))
sns.barplot(x="species", y="petal length (cm)", data=df, estimator="mean")
plt.title("Bar Chart: Average Petal Length per Species")
plt.xlabel("Species")
plt.ylabel("Average Petal Length (cm)")
plt.savefig("bar_chart.png")
plt.show()

# 3. Histogram - distribution of sepal length
plt.figure(figsize=(6,5))
plt.hist(df["sepal length (cm)"], bins=15, color="skyblue", edgecolor="black")
plt.title("Histogram: Sepal Length Distribution")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Frequency")
plt.savefig("histogram.png")
plt.show()

# 4. Scatter plot - sepal length vs petal length
plt.figure(figsize=(7,5))
sns.scatterplot(x="sepal length (cm)", y="petal length (cm)", hue="species", data=df, palette="Set1")
plt.title("Scatter Plot: Sepal Length vs Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend()
plt.savefig("scatter.png")
plt.show()

# Findings / Observations
print("\n📌 Findings:")
print("- The Iris dataset has no missing values and contains 150 samples with 4 numerical features.")
print("- Average petal length differs clearly among species (Setosa < Versicolor < Virginica).")
print("- Sepal length is normally distributed around ~5.8 cm.")
print("- Scatter plot shows Setosa is clearly separable from other species using petal length/width.")