# Machine Learning Project

This project demonstrates the implementation of various machine learning models including Multilayer Perceptron (MLP), LightGBM, and Convolutional Neural Networks (CNN) for both classification and regression tasks. It features preprocessing, feature engineering, with techniques such as PCA for dimensionality reduction, feature weighting based on correlation, and the innovative use of k-means clustering for feature enhancement.

Designed for flexibility, the framework allows easy switching between models and parameter adjustments to fit various datasets and objectives. It underscores the power of combining traditional machine learning models with deep learning techniques to achieve superior predictive performance.

## Project Structure

The project is structured into the following Python files:

1. `main.py`: Orchestrates the workflow, including data loading, preprocessing, model training, and evaluation.
2. `feature_engineering.py`: Handles data preprocessing and feature engineering.
3. `kmeans_cluster.py`: Implements k-means clustering for determining the optimal number of clusters.
4. `mlp.py`: Defines and trains a Multilayer Perceptron (MLP) model.
5. `lgb.py`: Defines and trains a LightGBM model.
6. `cnn.py`: Defines and trains a 1D Convolutional Neural Network (CNN) model.

## Dependencies

To run the project, ensure you have the following dependencies installed:

- Python 3.x
- NumPy
- Pandas
- scikit-learn
- TensorFlow
- Keras
- LightGBM

## Dataset Preparation

To utilize this project, you'll need to prepare your dataset as follows:

- Place your dataset within a `data` directory at the root of the project.
- Ensure your dataset is split into the following NumPy array files:
  - `train_data.npy`: Training data features.
  - `train_labels.npy`: Training data labels or targets.
  - `eval_data.npy`: Evaluation data features.
  - `eval_labels.npy`: Evaluation data labels or targets.

## Feature Engineering

`feature_engineering.py` includes functions for:

- Outlier removal based on z-scores.
- Data standardization for zero mean and unit variance.
- Dimensionality reduction using PCA for correlated features.
- Feature weighting based on correlation with the target variable.
- Adding k-means clustering labels as features.

## Models

The project includes:

Adjust `main.py` to set your desired parameters, such as task type (classification or regression) and models to use through switches (`classify`, `mlp_switch`, `lgb_switch`, `cnn_switch`).

1. **MLP (mlp.py)**: An MLP model for both tasks.
2. **LightGBM (lgb.py)**: A LightGBM model for both tasks.
3. **1D CNN (cnn.py)**: A 1D CNN model for both tasks.

Models are trained using k-fold cross-validation, selecting the best weights based on validation performance.

## Evaluation

Models are evaluated on an evaluation dataset, with **accuracy scores for classification** and **mean absolute errors for regression**.

## Model Flexibility

This project allows easy selection of machine learning models to be trained and evaluated through switches in `main.py` (`mlp_switch`, `lgb_switch`, `cnn_switch`). This design enables customization based on specific needs, computational resources, or preferences, showcasing the project's adaptability.

```python
# Model switches
mlp_switch = True  # Toggle MLP model
lgb_switch = True  # Toggle LightGBM model
cnn_switch = False  # Toggle CNN model

## License
This project is licensed under the MIT License - see the [LICENSE](https://opensource.org/licenses/MIT) for details.

