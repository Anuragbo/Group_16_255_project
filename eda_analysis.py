import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style for visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Load the dataset
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

print("="*80)
print("EXPLORATORY DATA ANALYSIS: Telco Customer Churn")
print("="*80)

# 1. Dataset Overview
print("\n1. DATASET OVERVIEW")
print("-" * 80)
print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"\nFirst few rows:")
print(df.head())

# 2. Data Types and Missing Values
print("\n2. DATA TYPES AND MISSING VALUES")
print("-" * 80)
print(f"\nData Types:\n{df.dtypes}")
print(f"\nMissing Values:\n{df.isnull().sum()}")
print(f"Missing Data Percentage:\n{(df.isnull().sum() / len(df) * 100).round(2)}")

# 3. Basic Statistics
print("\n3. DESCRIPTIVE STATISTICS")
print("-" * 80)
print(f"\nNumeric Columns Summary:")
print(df.describe())

# 4. Categorical Variables Summary
print("\n4. CATEGORICAL VARIABLES")
print("-" * 80)
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    print(f"\n{col}:")
    print(df[col].value_counts())
    print(f"Unique values: {df[col].nunique()}")

# 5. Target Variable Analysis (if Churn exists)
if 'Churn' in df.columns:
    print("\n5. TARGET VARIABLE ANALYSIS (Churn)")
    print("-" * 80)
    print(df['Churn'].value_counts())
    print(f"\nChurn Rate: {(df['Churn'].value_counts() / len(df) * 100).round(2)}%")

# 6. Numerical Columns Analysis
print("\n6. NUMERICAL COLUMNS DETAILED ANALYSIS")
print("-" * 80)
numeric_cols = df.select_dtypes(include=[np.number]).columns
print(f"Numeric columns: {list(numeric_cols)}")

# 7. Correlation Analysis
print("\n7. CORRELATION MATRIX")
print("-" * 80)
numeric_cols = df.select_dtypes(include=[np.number]).columns
if len(numeric_cols) > 0:
    corr_matrix = df[numeric_cols].corr()
    print(corr_matrix)

# 8. Visualizations
print("\n8. GENERATING VISUALIZATIONS...")
print("-" * 80)

# Distribution of numeric columns
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 10))
axes = axes.ravel()

numeric_cols_list = list(numeric_cols)
for idx, col in enumerate(numeric_cols_list[:4]):
    if idx < len(axes):
        axes[idx].hist(df[col], bins=30, edgecolor='black', alpha=0.7)
        axes[idx].set_title(f'Distribution of {col}')
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel('Frequency')

plt.tight_layout()
plt.savefig('01_numeric_distributions.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 01_numeric_distributions.png")
plt.close()

# Correlation heatmap
if len(df[numeric_cols].columns) > 0:
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('Correlation Matrix - Numeric Features')
    plt.tight_layout()
    plt.savefig('02_correlation_matrix.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: 02_correlation_matrix.png")
    plt.close()

# Categorical variables distribution
categorical_cols = df.select_dtypes(include=['object']).columns
num_categorical = len(categorical_cols)
if num_categorical > 0:
    fig, axes = plt.subplots(nrows=(num_categorical + 1) // 2, ncols=2, figsize=(14, 4 * ((num_categorical + 1) // 2)))
    axes = axes.ravel() if num_categorical > 1 else [axes]
    
    for idx, col in enumerate(categorical_cols):
        df[col].value_counts().plot(kind='bar', ax=axes[idx], edgecolor='black')
        axes[idx].set_title(f'Distribution of {col}')
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel('Count')
        axes[idx].tick_params(axis='x', rotation=45)
    
    # Hide extra subplots
    for idx in range(len(categorical_cols), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('03_categorical_distributions.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: 03_categorical_distributions.png")
    plt.close()

# Churn analysis if it exists
if 'Churn' in df.columns:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Churn distribution
    churn_counts = df['Churn'].value_counts()
    axes[0].bar(churn_counts.index, churn_counts.values, edgecolor='black', color=['green', 'red'])
    axes[0].set_title('Churn Distribution')
    axes[0].set_ylabel('Count')
    axes[0].set_xlabel('Churn')
    
    # Churn rate pie chart
    axes[1].pie(churn_counts.values, labels=churn_counts.index, autopct='%1.1f%%', startangle=90)
    axes[1].set_title('Churn Rate (%)')
    
    plt.tight_layout()
    plt.savefig('04_churn_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: 04_churn_analysis.png")
    plt.close()

print("\n" + "="*80)
print("EDA COMPLETE! Check the generated PNG files for visualizations.")
print("="*80)
