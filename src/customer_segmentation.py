import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import hdbscan
import gower

def load_data(filepath='data/curated.parquet'):
    df = pd.read_parquet(filepath)
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    return df, X, y

def apply_pca(X, variance_threshold=0.90):
    pca = PCA()
    pca.fit(X)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.argmax(cumulative_variance >= variance_threshold) + 1
    
    print(f"PCA Reduced dimensions from {X.shape[1]} to {n_components} (explaining {variance_threshold*100}% variance)")
    
    pca_optimal = PCA(n_components=n_components)
    X_pca = pca_optimal.fit_transform(X)
    
    # Plot explained variance
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--')
    plt.axhline(y=variance_threshold, color='r', linestyle='-')
    plt.axvline(x=n_components, color='r', linestyle='--')
    plt.title('PCA Cumulative Explained Variance')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.grid(True)
    plt.savefig('outputs/segmentation/pca_explained_variance.png')
    plt.close()
    
    return X_pca, n_components

def find_optimal_clusters(X_pca, max_k=10):
    silhouette_scores_kmeans = []
    silhouette_scores_gmm = []
    
    k_range = list(range(2, max_k + 1))
    
    for k in k_range:
        # KMeans
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels_kmeans = kmeans.fit_predict(X_pca)
        score_kmeans = silhouette_score(X_pca, labels_kmeans)
        silhouette_scores_kmeans.append(score_kmeans)
        
        # GMM
        gmm = GaussianMixture(n_components=k, random_state=42)
        labels_gmm = gmm.fit_predict(X_pca)
        score_gmm = silhouette_score(X_pca, labels_gmm)
        silhouette_scores_gmm.append(score_gmm)
        
    optimal_k_kmeans = k_range[np.argmax(silhouette_scores_kmeans)]
    optimal_k_gmm = k_range[np.argmax(silhouette_scores_gmm)]
    
    return optimal_k_kmeans, optimal_k_gmm, k_range, silhouette_scores_kmeans, silhouette_scores_gmm

def plot_unified_silhouette(k_range, scores_kmeans, scores_gmm, hdbscan_pca_score, hdbscan_gower_score):
    # Style 1: Line Plot Overlay
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, scores_kmeans, marker='o', label='K-Means (PCA)')
    plt.plot(k_range, scores_gmm, marker='s', label='GMM (PCA)')
    
    if hdbscan_pca_score is not None:
        plt.axhline(y=hdbscan_pca_score, color='green', linestyle='--', label=f'HDBSCAN (PCA) [{hdbscan_pca_score:.3f}]')
    if hdbscan_gower_score is not None:
        plt.axhline(y=hdbscan_gower_score, color='purple', linestyle='-.', label=f'HDBSCAN (Gower) [{hdbscan_gower_score:.3f}]')
        
    plt.title('Silhouette Scores (Line Overlay)')
    plt.xlabel('Number of Clusters (k) [Only applies to K-Means/GMM]')
    plt.ylabel('Silhouette Score')
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('outputs/segmentation/silhouette_scores_line.png')
    plt.close()
    
    # Style 2: Bar Chart showing the BEST score for each method
    methods = ['K-Means (PCA)', 'GMM (PCA)', 'HDBSCAN (PCA)', 'HDBSCAN (Gower)']
    best_scores = [max(scores_kmeans), max(scores_gmm), hdbscan_pca_score or 0, hdbscan_gower_score or 0]
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(methods, best_scores, color=['blue', 'orange', 'green', 'purple'])
    plt.title('Highest Achieved Silhouette Score by Algorithm')
    plt.ylabel('Silhouette Score')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.005, f'{yval:.3f}', ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig('outputs/segmentation/silhouette_scores_bar.png')
    plt.close()

def profile_segments(df, cluster_col, method_name):
    print(f"\n--- Profiling {method_name} ---")
    cols_to_profile = ['tenure', 'MonthlyCharges', 'Churn']
    contract_cols = [col for col in df.columns if 'Contract_' in col]
    cols_to_profile.extend(contract_cols)
    cols_to_profile = [c for c in cols_to_profile if c in df.columns]
    
    profile = df.groupby(cluster_col)[cols_to_profile].mean()
    profile['Cluster_Size'] = df.groupby(cluster_col).size()
    cols = ['Cluster_Size'] + [c for c in profile.columns if c != 'Cluster_Size']
    profile = profile[cols]
    
    print(profile)
    filename = method_name.lower().replace("-","").replace(" ","_").replace("(","").replace(")","")
    profile.to_csv(f'outputs/segmentation/{filename}_profile.csv')
    return profile

def calculate_churn_variance(profile, method_name):
    if len(profile) <= 1: return 0.0
    churn_variance = np.var(profile['Churn'])
    print(f"Churn Rate Variance for {method_name}: {churn_variance:.4f}")
    return churn_variance

def get_hdbscan_valid_silhouette(X, labels, metric='euclidean'):
    # Ignore noise points (-1) for silhouette calculation
    valid_indices = labels != -1
    if sum(valid_indices) <= 1 or len(set(labels[valid_indices])) < 2:
        return None
        
    X_valid = X[valid_indices]
    labels_valid = labels[valid_indices]
    
    if metric == 'precomputed':
        # X_valid must be the distance matrix of just the valid points
        X_valid = X_valid[:, valid_indices]
        
    return silhouette_score(X_valid, labels_valid, metric=metric)

def run_clustering_pipeline():
    os.makedirs('outputs/segmentation', exist_ok=True)
    print("Loading data...")
    df, X, y = load_data()
    
    print("Applying PCA...")
    X_pca, num_components = apply_pca(X, variance_threshold=0.90)
    
    print("Finding optimal K for K-Means and GMM...")
    best_k_kmeans, best_k_gmm, k_range, scores_kmeans, scores_gmm = find_optimal_clusters(X_pca, max_k=10)
    
    print("\nRunning best models...")
    df_results = df.copy()
    
    # Final KMeans
    kmeans = KMeans(n_clusters=best_k_kmeans, random_state=42, n_init=10)
    df_results['Cluster_KMeans'] = kmeans.fit_predict(X_pca)
    profile_kmeans = profile_segments(df_results, 'Cluster_KMeans', 'K-Means')
    var_kmeans = calculate_churn_variance(profile_kmeans, 'K-Means')
    
    # Final GMM
    gmm = GaussianMixture(n_components=best_k_gmm, random_state=42)
    df_results['Cluster_GMM'] = gmm.fit_predict(X_pca)
    profile_gmm = profile_segments(df_results, 'Cluster_GMM', 'GMM')
    var_gmm = calculate_churn_variance(profile_gmm, 'GMM')
    
    # HDBSCAN (PCA)
    print("\nRunning HDBSCAN (PCA Euclidean)...")
    hdb_pca = hdbscan.HDBSCAN(min_cluster_size=50)
    df_results['Cluster_HDBSCAN_PCA'] = hdb_pca.fit_predict(X_pca)
    num_hdb_pca = len(set(df_results['Cluster_HDBSCAN_PCA'])) - (1 if -1 in df_results['Cluster_HDBSCAN_PCA'] else 0)
    print(f"HDBSCAN (PCA) found {num_hdb_pca} clusters")
    
    sil_hdb_pca = get_hdbscan_valid_silhouette(X_pca, df_results['Cluster_HDBSCAN_PCA'].values, 'euclidean')
    profile_hdb_pca = profile_segments(df_results, 'Cluster_HDBSCAN_PCA', 'HDBSCAN (PCA)')
    valid_profile_hdb_pca = profile_hdb_pca.drop(-1, errors='ignore')
    var_hdb_pca = calculate_churn_variance(valid_profile_hdb_pca, 'HDBSCAN (PCA)')

    # Gower Distance Matrix
    print("\nComputing Gower Distance Matrix (this might take a moment)...")
    gower_mat = gower.gower_matrix(np.asarray(X)).astype(np.float64)
    
    # HDBSCAN (Gower)
    print("Running HDBSCAN (precomputed Gower)...")
    hdb_gower = hdbscan.HDBSCAN(metric='precomputed', min_cluster_size=50)
    df_results['Cluster_HDBSCAN_Gower'] = hdb_gower.fit_predict(gower_mat)
    num_hdb_gower = len(set(df_results['Cluster_HDBSCAN_Gower'])) - (1 if -1 in df_results['Cluster_HDBSCAN_Gower'] else 0)
    print(f"HDBSCAN (Gower) found {num_hdb_gower} clusters")
    
    sil_hdb_gower = get_hdbscan_valid_silhouette(gower_mat, df_results['Cluster_HDBSCAN_Gower'].values, 'precomputed')
    profile_hdb_gower = profile_segments(df_results, 'Cluster_HDBSCAN_Gower', 'HDBSCAN (Gower)')
    valid_profile_hdb_gower = profile_hdb_gower.drop(-1, errors='ignore')
    var_hdb_gower = calculate_churn_variance(valid_profile_hdb_gower, 'HDBSCAN (Gower)')
    
    print("\nPlotting unified Silhouette charts...")
    plot_unified_silhouette(k_range, scores_kmeans, scores_gmm, sil_hdb_pca, sil_hdb_gower)

    # Save final comparison
    with open('outputs/segmentation/churn_variance_comparison.txt', 'w') as f:
        f.write("Churn Rate Variance Across Clusters\n")
        f.write("-" * 35 + "\n")
        f.write(f"K-Means ({best_k_kmeans} clusters): {var_kmeans:.6f}\n")
        f.write(f"GMM ({best_k_gmm} clusters): {var_gmm:.6f}\n")
        f.write(f"HDBSCAN with PCA ({num_hdb_pca} clusters): {var_hdb_pca:.6f}\n")
        f.write(f"HDBSCAN with Gower ({num_hdb_gower} clusters): {var_hdb_gower:.6f}\n")
        f.write("\n* Higher variance indicates the clustering algorithm successfully separated high-churn users from low-churn users.")
        
    print("\nPipeline Complete! Results saved to outputs/segmentation/")
    
if __name__ == "__main__":
    run_clustering_pipeline()
