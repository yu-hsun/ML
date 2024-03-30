from feature_engineering import load_and_preprocess_data
from kmeans_cluster import kmeans_cluster
from mlp import mlp_train
from lgb import lgb_train
from cnn import cnn_train

from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error, silhouette_score, accuracy_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Choose the goal(classify or regression)
classify = True

# Model switches
mlp_switch = True
lgb_switch = True
cnn_switch = True


'''
# Try k-means
k, score = kmeans_cluster()
print(f'{k}: {score}')
'''

# Load post-feature engineering dataset
train_data, train_labels, eval_data, eval_labels = load_and_preprocess_data(k = 13) # Passing k = k into feature_engineering process 


# Model predictions storage
model_predictions = {}

if mlp_switch:
    preds_mlp = mlp_train(train_data, train_labels, eval_data, classify)
    model_predictions['mlp'] = preds_mlp

if lgb_switch:
    preds_lgb = lgb_train(train_data, train_labels, eval_data, classify)
    model_predictions['lgb'] = preds_lgb

if cnn_switch:
    preds_cnn = cnn_train(train_data, train_labels, eval_data, classify)
    model_predictions['cnn'] = preds_cnn



if classify:
    # Single model
    if mlp_switch:
        single_preds_mlp = np.argmax(preds_mlp, axis=1) # Select the index of the highest probability output
        mlp_test_accuracy = accuracy_score(eval_labels, single_preds_mlp)
        print(f"MLP Test Accuracy: {mlp_test_accuracy}")

    if cnn_switch:
        single_preds_cnn = np.argmax(preds_cnn, axis=1) # Select the index of the highest probability output
        cnn_test_accuracy = accuracy_score(eval_labels, single_preds_cnn)
        print(f"CNN Test Accuracy: {cnn_test_accuracy}")

    if lgb_switch:
        lgb_test_accuracy = accuracy_score(eval_labels, preds_lgb)
        print(f"LGB Test Accuracy: {lgb_test_accuracy}")

    # Two models
    if mlp_switch and lgb_switch and not cnn_switch: # MLP & LGB
        if mlp_test_accuracy > lgb_test_accuracy:
            print(f"MLP Test Accuracy: {mlp_test_accuracy}, LGB Test Accuracy: {lgb_test_accuracy}, The Better One is MLP")
        else:
            print(f"MLP Test Accuracy: {mlp_test_accuracy}, LGB Test Accuracy: {lgb_test_accuracy}, The Better One is LGB")
    
    if cnn_switch and lgb_switch and not mlp_switch: # CNN & LGB
        if cnn_test_accuracy > lgb_test_accuracy:
            print(f"CNN Test Accuracy: {cnn_test_accuracy}, LGB Test Accuracy: {lgb_test_accuracy}, The Better One is CNN")
        else:
            print(f"CNN Test Accuracy: {cnn_test_accuracy}, LGB Test Accuracy: {lgb_test_accuracy}, The Better One is LGB")

    if mlp_switch and cnn_switch: # MLP & CNN
        # Combine CNN & MLP predictions by averaging
        combined_preds = (preds_mlp + preds_cnn) / 2
        mlp_cnn_preds = np.argmax(combined_preds, axis=1)
        mlp_cnn_test_accuracy = accuracy_score(eval_labels, mlp_cnn_preds)
        print(f"MLP Test Accuracy: {mlp_test_accuracy}, CNN Test Accuracy: {cnn_test_accuracy}, MLP&CNN(Combined Models) Test Accuracy: {mlp_cnn_test_accuracy}")

    # Combine three models by voting
    if mlp_switch and lgb_switch and cnn_switch:
        # Determine the best model based on validation accuracy
        validation_accuracies = {
            'mlp': mlp_test_accuracy,
            'cnn': cnn_test_accuracy,    
            'lgb': lgb_test_accuracy   
        }
        best_model = max(validation_accuracies, key=validation_accuracies.get)

        final_preds = []
        for i in range(len(preds_mlp)):
            votes = [single_preds_mlp[i], single_preds_cnn[i], preds_lgb[i]]
            vote_result = max(set(votes), key=votes.count)
            
            # Check if there is a majority vote
            if votes.count(vote_result) > 1:
                final_preds.append(vote_result)

            else:
                # Dynamically choose the best model
                if best_model == 'mlp':
                    final_preds.append(single_preds_mlp[i])
                elif best_model == 'cnn':
                    final_preds.append(single_preds_cnn[i])
                elif best_model == 'lgb':
                    final_preds.append(preds_lgb[i])

        final_preds = np.array(final_preds)
        test_accuracy = accuracy_score(eval_labels, final_preds)
        print(f"Test Accuracy: {test_accuracy}")


else:
    predictions_list = []

    # Single model
    for model_name, preds in model_predictions.items():
        mae = mean_absolute_error(eval_labels, preds)
        predictions_list.append(preds)
        print(f"{model_name} Test MAE: {mae}")

    # Combine models
    if len(predictions_list) > 1:
        preds_combined = np.mean(predictions_list, axis=0)
        
        # Evaluating the combined predictions
        combined_mae = mean_absolute_error(eval_labels, preds_combined)
        combined_mse = mean_squared_error(eval_labels, preds_combined)
        combined_r2 = r2_score(eval_labels, preds_combined)
        combined_mape = mean_absolute_percentage_error(eval_labels, preds_combined)

        print(f"Combined Test MAE: {combined_mae}, Combined Test MSE: {combined_mse}, Combined Test R^2: {combined_r2}, Combined Test MAPE: {combined_mape}")




