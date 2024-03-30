from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from scipy import stats
import pandas as pd
import numpy as np

def load_and_preprocess_data(k):
    # Load data
    train_data = np.load('data/train_data.npy')
    train_labels = np.load('data/train_labels.npy')
    eval_data = np.load('data/eval_data.npy')
    eval_labels = np.load('data/eval_labels.npy')

    # Drop outliers
    z_scores = np.abs(stats.zscore(train_data))
    outliers = (z_scores > 4.4).any(axis=1)  # Adjust the threshold as necessary e.g:4.45
    train_data = train_data[~outliers]
    train_labels = train_labels[~outliers]
    print(train_data.shape)

    # Z-standardize the data
    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    eval_data = scaler.transform(eval_data) 

    # PCA PROCESS
    df_train_data = pd.DataFrame(train_data)

    # Calculate the correlation matrix. Use absolute value to consider strong negative correlations as well
    corr_matrix = df_train_data.corr().abs()  

    # Identify groups of highly correlated features, Set the threshold for high correlation
    high_corr_threshold = 0.9  # Adjust here
    highly_correlated_groups = []

    # Finds indices of highly correlated pairs, avoiding self-correlation
    for i in range(corr_matrix.shape[0]):
        for j in range(i+1, corr_matrix.shape[1]):
            if corr_matrix.iloc[i, j] >= high_corr_threshold:
                found = False
                for group in highly_correlated_groups:
                    if i in group or j in group:
                        group.add(i)
                        group.add(j)
                        found = True
                        break
                if not found:
                    highly_correlated_groups.append({i, j})
    #print(highly_correlated_groups)
    df_train_data = pd.DataFrame(train_data)


    original_feature_correlations = {}

    for group in highly_correlated_groups:
        for feature_idx in group:
            # Calculate correlation of this feature with the labels
            correlation, _ = stats.pearsonr(df_train_data.iloc[:, feature_idx], train_labels)
            abs_correlation = abs(correlation)

            original_feature_correlations[feature_idx] = abs_correlation


    df_train = pd.DataFrame(train_data)
    df_eval = pd.DataFrame(eval_data)

    # Initialize storage for PCA-transformed features for both datasets
    train_pca_transformed_features = []
    eval_pca_transformed_features = []

    adjust_highly_correlated_groups = [[] for _ in range(len(highly_correlated_groups))]
    pca_corr = {}

    for group_idx, feature_indices in enumerate(highly_correlated_groups):
        train_feature_subset = df_train.iloc[:, list(feature_indices)]
        eval_feature_subset = df_eval.iloc[:, list(feature_indices)]
        

        # Set the number of PCA components dynamically based on the size of the group 
        n_components = min(len(feature_indices), 25) # Adjust 25 for the upperbond 
        pca = PCA(n_components = n_components)
        
        pca.fit(train_feature_subset)
        train_pca_features = pca.transform(train_feature_subset)
        eval_pca_features = pca.transform(eval_feature_subset)

        for i, feature_index in enumerate(feature_indices):
            correlation, _ = stats.pearsonr(train_pca_features[:, i], train_labels)
            pca_corr[feature_index] = abs(correlation)

            if pca_corr[feature_index] > original_feature_correlations[feature_index]:
                adjust_highly_correlated_groups[group_idx].append(feature_index)

                train_pca_transformed_features.append(train_pca_features[:, i].reshape(-1, 1))
                eval_pca_transformed_features.append(eval_pca_features[:, i].reshape(-1, 1))


    # Remove the original features from the datasets
    for feature_indices in adjust_highly_correlated_groups:
        df_train.drop(df_train.columns[list(feature_indices)], axis=1, inplace=True)
        df_eval.drop(df_eval.columns[list(feature_indices)], axis=1, inplace=True)


    # Flatten the list of PCA-transformed features for concatenation
    train_pca_transformed_features = np.hstack(train_pca_transformed_features)
    eval_pca_transformed_features = np.hstack(eval_pca_transformed_features)
    train_pca_df = pd.DataFrame(train_pca_transformed_features, index=df_train.index)
    eval_pca_df = pd.DataFrame(eval_pca_transformed_features, index=df_eval.index)

    # Concatenate the original data (with highly correlated features removed) with the new PCA features
    df_train = pd.concat([df_train.reset_index(drop=True), train_pca_df.reset_index(drop=True)], axis=1)
    df_eval = pd.concat([df_eval.reset_index(drop=True), eval_pca_df.reset_index(drop=True)], axis=1)


    train_data = df_train.values
    eval_data = df_eval.values

    # Feature weighting
    feature_weight = {}
    for i in range(train_data.shape[1]):  
        correlation, _ = stats.pearsonr(train_data[:, i], train_labels)
        feature_weight[f'feature{i+1}'] = abs(correlation) 

    # Standerdizing feature weight
    total_weight = sum(feature_weight.values())
    standardized_weights = {feature: weight / total_weight for feature, weight in feature_weight.items()}

    # Adding weight to the data
    train_data = np.hstack([train_data[:, i:i+1] * weight for i, weight in enumerate(standardized_weights.values())])
    eval_data = np.hstack([eval_data[:, i:i+1] * weight for i, weight in enumerate(standardized_weights.values())])


    # Add k-means clustering group as a new feature
    n_clusters = k

    # Initialize KMeans with the desired number of clusters and fit it to the weighted training data
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(train_data)
    # Predict clusters for the weighted training and evaluation datasets
    train_clusters = kmeans.predict(train_data).reshape(-1, 1)
    eval_clusters = kmeans.predict(eval_data).reshape(-1, 1)

    # OneHot encoder
    encoder = OneHotEncoder()
    encoder.fit(train_clusters)

    train_clusters_onehot = encoder.transform(train_clusters).toarray()
    eval_clusters_onehot = encoder.transform(eval_clusters).toarray()

    # Append the cluster labels as features
    train_data_with_clusters = np.hstack((train_data, train_clusters_onehot))
    eval_data_with_clusters = np.hstack((eval_data, eval_clusters_onehot))
    train_data = train_data_with_clusters
    eval_data = eval_data_with_clusters

    print(train_data.shape, eval_data.shape)

    return train_data, train_labels, eval_data, eval_labels