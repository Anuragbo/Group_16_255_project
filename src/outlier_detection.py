import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from scipy.stats import pointbiserialr

def load_clustered_data(filepath='data/clustered.parquet'):
    """Loads the dataset containing features, Churn label, and cluster assignments."""
    df = pd.read_parquet(filepath)
    # The feature columns are those purely not associated with labels or clusters
    non_feature_cols = ['Churn'] + [col for col in df.columns if col.startswith('Cluster_')]
    X = df.drop(columns=non_feature_cols)
    return df, X, non_feature_cols

def apply_outlier_detection(X, df, contamination=0.02):
    """
    Applies Isolation Forest and Local Outlier Factor to the data.
    Returns anomaly scores and predicted outlier labels.
    """
    results = df.copy()
    
    # 1. Isolation Forest
    # Lower isolation score = more abnormal
    iso = IsolationForest(contamination=contamination, random_state=42)
    results['IsoForest_Label'] = iso.fit_predict(X) 
    # decision_function gives score where lower is more anomalous
    results['IsoForest_Score'] = iso.decision_function(X) 
    
    # 2. Local Outlier Factor
    # For LOF, negative outlier factor is used (lower = more abnormal)
    lof = LocalOutlierFactor(n_neighbors=20, contamination=contamination)
    results['LOF_Label'] = lof.fit_predict(X)
    results['LOF_Score'] = lof.negative_outlier_factor_
    
    # Let's align on a unified 'Anomaly Score' convention where HIGHER = MORE ANOMALOUS.
    # So we invert the scores.
    results['IsoForest_Anomaly_Score'] = -results['IsoForest_Score']
    results['LOF_Anomaly_Score'] = -results['LOF_Score']
    
    return results

def extract_top_anomalies(results, X, k=50):
    """Extracts top-k most anomalous profiles for case studies."""
    # Top K for Isolation Forest
    top_iso = results.nlargest(k, 'IsoForest_Anomaly_Score')
    # Top K for LOF
    top_lof = results.nlargest(k, 'LOF_Anomaly_Score')
    
    # Calculate profile (mean values) of these outliers compared to the rest of the dataset
    global_mean = X.mean()
    iso_mean = top_iso[X.columns].mean()
    lof_mean = top_lof[X.columns].mean()
    
    profile_df = pd.DataFrame({
        'Global_Average': global_mean,
        'Top_IsoForest_Average': iso_mean,
        'Top_LOF_Average': lof_mean
    })
    
    profile_df['Iso_Pct_Difference'] = ((profile_df['Top_IsoForest_Average'] - profile_df['Global_Average']) / (profile_df['Global_Average'].replace(0, 1e-9))) * 100
    profile_df['LOF_Pct_Difference'] = ((profile_df['Top_LOF_Average'] - profile_df['Global_Average']) / (profile_df['Global_Average'].replace(0, 1e-9))) * 100
    
    return profile_df, top_iso, top_lof

def boxplot_anomaly_vs_churn(results, score_col, title, filename):
    plt.figure(figsize=(8, 6))
    
    # Split data by Churn
    churn_0 = results[results['Churn'] == 0][score_col]
    churn_1 = results[results['Churn'] == 1][score_col]
    
    plt.boxplot([churn_0, churn_1], labels=['0 (No)', '1 (Yes)'], patch_artist=True,
                boxprops=dict(facecolor='lightblue', color='blue'),
                medianprops=dict(color='red', linewidth=2))
                
    plt.title(f'{title} vs. Churn Label')
    plt.xlabel('Churn')
    plt.ylabel('Anomaly Score (Higher is more anomalous)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'outputs/outlier_detection/{filename}')
    plt.close()

def analyze_churn_correlation(results):
    """Examines correlation between anomaly scores and churn label."""
    # Point-Biserial Correlation (continuous var vs binary var)
    iso_corr, iso_pval = pointbiserialr(results['Churn'], results['IsoForest_Anomaly_Score'])
    lof_corr, lof_pval = pointbiserialr(results['Churn'], results['LOF_Anomaly_Score'])
    
    # Store results
    corr_results = pd.DataFrame({
        'Method': ['Isolation Forest', 'LOF'],
        'Point_Biserial_Correlation': [iso_corr, lof_corr],
        'P-Value': [iso_pval, lof_pval]
    })
    
    corr_results.to_csv('outputs/outlier_detection/anomaly_churn_correlation.csv', index=False)
    
    boxplot_anomaly_vs_churn(results, 'IsoForest_Anomaly_Score', 'Isolation Forest Anomaly Score', 'boxplot_isoforest_churn.png')
    boxplot_anomaly_vs_churn(results, 'LOF_Anomaly_Score', 'LOF Anomaly Score', 'boxplot_lof_churn.png')
    
    return corr_results

def cross_reference_clusters(top_anomalies, method_name, cluster_cols):
    """Answers: Which clusters are producing the most extreme outliers?"""
    cross_tabs = {}
    for col in cluster_cols:
        if col in top_anomalies.columns:
            counts = top_anomalies[col].value_counts().reset_index()
            counts.columns = [col, 'Outlier_Count']
            
            # Save cross tab
            counts.to_csv(f'outputs/outlier_detection/{method_name}_{col}_overlap.csv', index=False)
            cross_tabs[col] = counts
            
            # Plot
            plt.figure(figsize=(8, 5))
            plt.bar(counts[col].astype(str), counts['Outlier_Count'], color='teal')
            plt.title(f'Cluster Mapping of Top Anomalies ({method_name})')
            plt.xlabel(f'Cluster ID ({col})')
            plt.ylabel('Number of Anomalies')
            plt.grid(axis='y', linestyle='--', alpha=0.6)
            plt.tight_layout()
            plt.savefig(f'outputs/outlier_detection/plot_{method_name}_{col}_distribution.png')
            plt.close()
            
    return cross_tabs

def run_outlier_detection_pipeline():
    os.makedirs('outputs/outlier_detection', exist_ok=True)
    
    print("Loading clustered data...")
    df, X, non_feature_cols = load_clustered_data()
    
    print("Applying Isolation Forest and LOF...")
    results = apply_outlier_detection(X, df, contamination=0.03)  # top 3% as outliers
    
    print("Extracting top-k anomalous profiles...")
    profile_df, top_iso, top_lof = extract_top_anomalies(results, X, k=100)
    profile_df.to_csv('outputs/outlier_detection/anomalous_feature_profiles.csv')
    
    print("Analyzing anomaly correlation with Churn...")
    corr_res = analyze_churn_correlation(results)
    print("\nAnomaly-Churn Correlation:")
    print(corr_res)
    
    # Save the detailed top anomalies list
    cols_to_view = ['Churn', 'IsoForest_Anomaly_Score', 'LOF_Anomaly_Score'] + [c for c in non_feature_cols if 'Cluster' in c]
    top_iso[cols_to_view].to_csv('outputs/outlier_detection/top_100_isolation_forest.csv', index=False)
    top_lof[cols_to_view].to_csv('outputs/outlier_detection/top_100_lof.csv', index=False)
    
    print("\nCross-referencing outliers with cluster membership...")
    cluster_cols = [c for c in non_feature_cols if c.startswith('Cluster_')]
    cross_reference_clusters(top_iso, 'IsoForest', cluster_cols)
    cross_reference_clusters(top_lof, 'LOF', cluster_cols)
    
    print("\nSection 8 Outlier Detection Pipeline Complete!")
    print("Outputs successfully saved to outputs/outlier_detection/")

if __name__ == "__main__":
    run_outlier_detection_pipeline()
