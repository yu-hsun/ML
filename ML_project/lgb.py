from sklearn.model_selection import KFold

import numpy as np
import lightgbm as lgb


def lgb_train(train_data, train_labels, eval_data, classify):

    if classify:
        num_classes = len(np.unique(train_labels))
        lgb_params = {
            'objective': 'multiclass',
            'metric': 'multi_logloss',
            'num_class': num_classes,
            'boosting_type': 'gbdt',
            'n_estimators': 7000,
            'num_leaves': 128,
            'learning_rate': 0.01,
            'feature_fraction': 0.9,
            'verbosity': -1
        }

        n_folds = 5

        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

        for train_idx, test_idx in kf.split(train_data, train_labels):
            X_train, X_test = train_data[train_idx], train_data[test_idx]
            y_train, y_test = train_labels[train_idx], train_labels[test_idx]

            # Initialize the LightGBM model
            lgb_model = lgb.LGBMClassifier(**lgb_params)

            # Train the LightGBM model with early stopping
            lgb_model.fit(X_train, y_train, 
                        eval_set=[(X_train, y_train), (X_test, y_test)], 
                        eval_metric='multi_logloss',
                        callbacks=[
                            lgb.callback.early_stopping(stopping_rounds=100),
                            lgb.callback.log_evaluation(period=100)
                        ],
                        )

        preds = lgb_model.predict(eval_data, num_iteration=lgb_model.best_iteration_)

        return preds
    
#-----------------------------------------------non-classification model-----------------------------------------------
    else:
        lgb_params = {
        'objective': 'regression',
        'metric': 'mae',
        'boosting_type': 'gbdt',
        'n_estimators': 7000,
        'num_leaves': 128,
        'learning_rate': 0.01,
        'feature_fraction': 0.9,
        'verbosity': -1,
    }

    n_folds = 5

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    for train_idx, test_idx in kf.split(train_data, train_labels):
        X_train, X_test = train_data[train_idx], train_data[test_idx]
        y_train, y_test = train_labels[train_idx], train_labels[test_idx]

        # Initialize the LightGBM model
        lgb_model = lgb.LGBMRegressor(**lgb_params)

        # Train the LightGBM model with early stopping
        lgb_model.fit(X_train, y_train, 
                    eval_set=[(X_train, y_train), (X_test, y_test)], 
                    eval_metric='mae',
                    callbacks=[
                        lgb.callback.early_stopping(stopping_rounds=100),
                        lgb.callback.log_evaluation(period=100)
                    ],
                    )

    preds = lgb_model.predict(eval_data, num_iteration=lgb_model.best_iteration_)

    return preds
        

