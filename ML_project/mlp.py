from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, mean_absolute_error
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout, ReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from tensorflow.keras.regularizers import l2
import os
import numpy as np
import tensorflow as tf
import shutil 

# Function to clear out the old model
def clear_checkpoint_directory(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)

def mlp_train(train_data, train_labels, eval_data, classify):

    # Clear out the old models before starting the new training session
    directory = 'NN_Models/'
    clear_checkpoint_directory(directory)

    if classify:
        def create_mlp(num_continuous_features, num_labels, hidden_units, dropout_rates, learning_rate):
            # Input layer for continuous features
            input_continuous = Input(shape=(num_continuous_features,))

            # Initial processing
            x = BatchNormalization()(input_continuous)
            x = Dropout(dropout_rates[0])(x)

            # Hidden layers
            for units, dropout_rate in zip(hidden_units, dropout_rates[1:]):
                x = Dense(units, kernel_initializer='he_normal')(x)
                x = BatchNormalization()(x)
                x = ReLU()(x)
                x = Dropout(dropout_rate)(x)

            # Output layer
            out = Dense(num_labels, activation='softmax')(x)

            # Model assembly
            model = Model(inputs=input_continuous, outputs=out)
            model.compile(optimizer=Adam(learning_rate=learning_rate),
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy'])

            return model


        # Define model training parameters
        batch_size = 64
        hidden_units = [128, 128, 128] 
        dropout_rates = [0.1, 0.1, 0.1]
        learning_rate = 0.01
        num_continuous_features = train_data.shape[1]
        num_labels = len(np.unique(train_labels))

        # Adjust the directory for saving models as needed
        directory = 'NN_Models/'
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Folds
        num_folds = 5
        kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

        # Initialize variables for storing the best model weights and performance metrics
        best_model_weights = None
        best_accuracy = 0

        for fold, (train_idx, test_idx) in enumerate(kf.split(train_data, train_labels)):
            print(f"Starting fold {fold + 1}")

            # Split data into training and validation sets for the current fold
            X_train, X_val = train_data[train_idx], train_data[test_idx]
            y_train, y_val = train_labels[train_idx], train_labels[test_idx]

            model = create_mlp(num_continuous_features, num_labels, hidden_units, dropout_rates, learning_rate)

            # Callbacks for training
            ckp_path = os.path.join(directory, f'best_model_fold_{fold + 1}.weights.h5')
            rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, min_delta=1e-4, mode='min')
            ckp = ModelCheckpoint(ckp_path, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='min')
            es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1, restore_best_weights=True)

            # Model fitting
            model.fit(X_train, y_train, validation_data=(X_val, y_val),
                    epochs=100, batch_size=batch_size,
                    callbacks=[rlr, ckp, es], verbose=2)
            model.load_weights(ckp_path) # Load the best model weights for the current fold


            # Evaluate the model on the validation set of the current fold
            val_preds = model.predict(X_val)
            val_accuracy = accuracy_score(y_val, np.argmax(val_preds, axis=1))

            # Update the best model weights and performance metrics if the current fold's model is better
            if val_accuracy > best_accuracy:
                best_model_weights = model.get_weights()
                best_accuracy = val_accuracy

            tf.keras.backend.clear_session()

        # Load the best model weights and evaluate on the test set
        model = create_mlp(num_continuous_features, num_labels, hidden_units, dropout_rates, learning_rate)
        model.set_weights(best_model_weights)

        preds = model.predict(eval_data)

        #return np.argmax(preds, axis=1)
        return preds
    
#-----------------------------------------------non-classification model-----------------------------------------------
    if not classify:

        def create_mlp(num_continuous_features, num_labels, hidden_units, dropout_rates, learning_rate, l2_strength=0.01):
            # Input layer for continuous features
            input_continuous = Input(shape=(num_continuous_features,))

            # Initial processing
            x = BatchNormalization()(input_continuous)
            x = Dropout(dropout_rates[0])(x)

            # Hidden layers
            for units, dropout_rate in zip(hidden_units, dropout_rates[1:]):
                x = Dense(units, kernel_regularizer=l2(l2_strength), kernel_initializer='he_normal')(x)
                x = BatchNormalization()(x)
                x = ReLU()(x)
                x = Dropout(dropout_rate)(x)

            # Output layer
            out = Dense(num_labels, kernel_regularizer=l2(l2_strength), kernel_initializer='he_normal')(x)

            # Model assembly
            model = Model(inputs=input_continuous, outputs=out)
            model.compile(optimizer=Adam(learning_rate=learning_rate),
                        loss='mean_absolute_error',
                        metrics=['mean_absolute_error', 'mean_squared_error'])

            return model


        # Define model training parameters
        batch_size = 64
        hidden_units = [128, 128, 128] 
        dropout_rates = [0.1, 0.1, 0.1]
        learning_rate = 0.01
        num_continuous_features = train_data.shape[1]  
        num_labels = 1  

        # Adjust the directory for saving models as needed
        directory = 'NN_Models/'
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Folds
        num_folds = 5
        kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

        # Initialize variables for storing the best model weights and performance metrics
        best_model_weights = None
        best_mae = float('inf')

        for fold, (train_idx, test_idx) in enumerate(kf.split(train_data, train_labels)):
            print(f"Starting fold {fold + 1}")

            # Split data into training and validation sets for the current fold
            X_train, X_val = train_data[train_idx], train_data[test_idx]
            y_train, y_val = train_labels[train_idx], train_labels[test_idx]

            model = create_mlp(num_continuous_features, num_labels, hidden_units, dropout_rates, learning_rate)

            # Callbacks for training
            ckp_path = os.path.join(directory, f'best_model_fold_{fold + 1}.weights.h5')
            rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, min_delta=1e-4, mode='min')
            ckp = ModelCheckpoint(ckp_path, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='min')
            es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1, restore_best_weights=True)

            # Model fitting
            model.fit(X_train, y_train, validation_data=(X_val, y_val),
                    epochs=100, batch_size=batch_size,
                    callbacks=[rlr, ckp, es], verbose=2)
            model.load_weights(ckp_path)

            # Evaluate the model on the validation set of the current fold
            val_preds = model.predict(X_val)
            val_mae = mean_absolute_error(y_val, val_preds)

            # Update the best model weights and performance metrics if the current fold's model is better
            if val_mae < best_mae:
                best_model_weights = model.get_weights()
                best_mae = val_mae

            tf.keras.backend.clear_session()

        # Load the best model weights and evaluate on the test set
        model = create_mlp(num_continuous_features, num_labels, hidden_units, dropout_rates, learning_rate)
        model.set_weights(best_model_weights)

        preds = model.predict(eval_data)

        return preds

