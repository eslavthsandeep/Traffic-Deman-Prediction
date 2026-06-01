# """
# =============================================================================
# Model Training & Ensembling - Traffic Demand Prediction (Flipkart GRID)
# =============================================================================
# Loads pre-extracted features from 'dataset/train_features.csv' and 'dataset/test_features.csv'.
# Trains 4 models (LightGBM, XGBoost, CatBoost, ExtraTrees) with 5-Fold Cross-Validation.
# Searches for the optimal ensemble weights and outputs:
# - Final ensemble submission: 'output/submission.csv'
# - Results summary: 'output/results_summary.json'
# - Feature importance plot: 'output/lgb_feature_importance.png'
# =============================================================================
# """

# import pandas as pd
# import numpy as np
# import os
# import json
# import warnings
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt

# # Modeling
# from sklearn.model_selection import KFold
# from sklearn.metrics import r2_score
# from sklearn.ensemble import ExtraTreesRegressor

# warnings.filterwarnings('ignore')

# # Boosting libraries with fallbacks
# try:
#     from xgboost import XGBRegressor
#     HAS_XGB = True
# except ImportError:
#     HAS_XGB = False
#     print("[WARN] XGBoost not available. Will skip XGBoost.")

# try:
#     from lightgbm import LGBMRegressor
#     import lightgbm as lgb
#     HAS_LGB = True
# except ImportError:
#     HAS_LGB = False
#     print("[WARN] LightGBM not available. Will skip LightGBM.")

# try:
#     from catboost import CatBoostRegressor
#     HAS_CAT = True
# except ImportError:
#     HAS_CAT = False
#     print("[WARN] CatBoost not available. Will skip CatBoost.")

# # Configuration
# DATA_DIR = "dataset"
# OUTPUT_DIR = "output"
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# RANDOM_STATE = 42
# N_FOLDS = 5
# np.random.seed(RANDOM_STATE)

# def main():
#     print("=" * 70)
#     print("STEP 1: LOADING FEATURE-EXTRACTED DATASETS")
#     print("=" * 70)
    
#     train_features_path = os.path.join(DATA_DIR, "train_features.csv")
#     test_features_path = os.path.join(DATA_DIR, "test_features.csv")
    
#     if not os.path.exists(train_features_path) or not os.path.exists(test_features_path):
#         raise FileNotFoundError(
#             "Feature-extracted CSVs not found! Please run 'extract_features.py' first."
#         )
        
#     train_df = pd.read_csv(train_features_path)
#     test_df = pd.read_csv(test_features_path)
    
#     print(f"Loaded Train Features shape: {train_df.shape}")
#     print(f"Loaded Test Features shape:  {test_df.shape}")
    
#     # Identify target and feature columns
#     target_col = 'demand'
#     exclude_cols = ['demand', 'Index']
#     feature_cols = [c for c in train_df.columns if c not in exclude_cols]
    
#     print(f"Total features: {len(feature_cols)}")
    
#     # Prepare NumPy arrays for modeling
#     X_train = train_df[feature_cols].values.astype(np.float64)
#     y_train = train_df[target_col].values.astype(np.float64)
#     X_test = test_df[feature_cols].values.astype(np.float64)
#     test_indices = test_df['Index'].values.astype(int)
    
#     # Replace any remaining NaNs or Infs
#     X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
#     X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
    
#     print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
#     print(f"X_test shape:  {X_test.shape}")
    
#     print("\n" + "=" * 70)
#     print("STEP 2: MODEL TRAINING WITH 5-FOLD CROSS-VALIDATION")
#     print("=" * 70)
    
#     kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
#     results = {}
#     oof_predictions = {}
#     test_predictions = {}
    
#     # ---- 2.1 LightGBM ----
#     if HAS_LGB:
#         print("\n--- Training LightGBM ---")
#         lgb_params = {
#             'n_estimators': 2000,
#             'learning_rate': 0.03,
#             'max_depth': 8,
#             'num_leaves': 63,
#             'min_child_samples': 20,
#             'subsample': 0.8,
#             'colsample_bytree': 0.8,
#             'reg_alpha': 0.1,
#             'reg_lambda': 1.0,
#             'random_state': RANDOM_STATE,
#             'n_jobs': -1,
#             'verbose': -1
#         }
        
#         lgb_oof = np.zeros(len(X_train))
#         lgb_test_preds = np.zeros(len(X_test))
#         lgb_scores = []
        
#         for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
#             X_tr, X_val = X_train[train_idx], X_train[val_idx]
#             y_tr, y_val = y_train[train_idx], y_train[val_idx]
            
#             model = LGBMRegressor(**lgb_params)
#             model.fit(
#                 X_tr, y_tr,
#                 eval_set=[(X_val, y_val)],
#                 callbacks=[
#                     lgb.early_stopping(100, verbose=False),
#                     lgb.log_evaluation(0)
#                 ]
#             )
            
#             val_pred = model.predict(X_val)
#             lgb_oof[val_idx] = val_pred
#             lgb_test_preds += model.predict(X_test) / N_FOLDS
            
#             fold_r2 = r2_score(y_val, val_pred)
#             lgb_scores.append(fold_r2)
#             print(f"  Fold {fold+1}: R2 = {fold_r2:.6f}")
            
#         overall_r2 = r2_score(y_train, lgb_oof)
#         print(f"  LightGBM OOF R2: {overall_r2:.6f}")
#         print(f"  LightGBM Mean CV R2: {np.mean(lgb_scores):.6f} +/- {np.std(lgb_scores):.6f}")
        
#         results['LightGBM'] = {'mean_r2': np.mean(lgb_scores), 'std_r2': np.std(lgb_scores), 'oof_r2': overall_r2}
#         oof_predictions['LightGBM'] = lgb_oof
#         test_predictions['LightGBM'] = lgb_test_preds
        
#         # Save feature importance plot using LGBM
#         fi = pd.DataFrame({
#             'feature': feature_cols,
#             'importance': model.feature_importances_
#         }).sort_values('importance', ascending=False)
        
#         print("\n  Top 10 Features (LightGBM):")
#         print(fi.head(10).to_string(index=False))
        
#         fig, ax = plt.subplots(figsize=(10, 10))
#         fi_top = fi.head(25)
#         ax.barh(range(len(fi_top)), fi_top['importance'].values, color='steelblue')
#         ax.set_yticks(range(len(fi_top)))
#         ax.set_yticklabels(fi_top['feature'].values)
#         ax.invert_yaxis()
#         ax.set_title('LightGBM Feature Importance (Top 25)', fontsize=14)
#         ax.set_xlabel('Importance')
#         plt.tight_layout()
#         fi_path = os.path.join(OUTPUT_DIR, 'lgb_feature_importance.png')
#         plt.savefig(fi_path, dpi=150, bbox_inches='tight')
#         plt.close()
#         print(f"  [SAVED] {fi_path}")
        
#     # ---- 2.2 XGBoost ----
#     if HAS_XGB:
#         print("\n--- Training XGBoost ---")
#         xgb_params = {
#             'n_estimators': 2000,
#             'learning_rate': 0.03,
#             'max_depth': 7,
#             'min_child_weight': 5,
#             'subsample': 0.8,
#             'colsample_bytree': 0.8,
#             'reg_alpha': 0.1,
#             'reg_lambda': 1.0,
#             'tree_method': 'hist',
#             'random_state': RANDOM_STATE,
#             'n_jobs': -1,
#             'verbosity': 0
#         }
        
#         xgb_oof = np.zeros(len(X_train))
#         xgb_test_preds = np.zeros(len(X_test))
#         xgb_scores = []
        
#         for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
#             X_tr, X_val = X_train[train_idx], X_train[val_idx]
#             y_tr, y_val = y_train[train_idx], y_train[val_idx]
            
#             model = XGBRegressor(**xgb_params)
#             model.fit(
#                 X_tr, y_tr,
#                 eval_set=[(X_val, y_val)],
#                 verbose=False
#             )
            
#             val_pred = model.predict(X_val)
#             xgb_oof[val_idx] = val_pred
#             xgb_test_preds += model.predict(X_test) / N_FOLDS
            
#             fold_r2 = r2_score(y_val, val_pred)
#             xgb_scores.append(fold_r2)
#             print(f"  Fold {fold+1}: R2 = {fold_r2:.6f}")
            
#         overall_r2 = r2_score(y_train, xgb_oof)
#         print(f"  XGBoost OOF R2: {overall_r2:.6f}")
#         print(f"  XGBoost Mean CV R2: {np.mean(xgb_scores):.6f} +/- {np.std(xgb_scores):.6f}")
        
#         results['XGBoost'] = {'mean_r2': np.mean(xgb_scores), 'std_r2': np.std(xgb_scores), 'oof_r2': overall_r2}
#         oof_predictions['XGBoost'] = xgb_oof
#         test_predictions['XGBoost'] = xgb_test_preds
        
#     # ---- 2.3 CatBoost ----
#     if HAS_CAT:
#         print("\n--- Training CatBoost ---")
#         cat_params = {
#             'iterations': 2000,
#             'learning_rate': 0.05,
#             'depth': 8,
#             'l2_leaf_reg': 3,
#             'random_seed': RANDOM_STATE,
#             'verbose': 0,
#             'early_stopping_rounds': 100,
#         }
        
#         cat_oof = np.zeros(len(X_train))
#         cat_test_preds = np.zeros(len(X_test))
#         cat_scores = []
        
#         for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
#             X_tr, X_val = X_train[train_idx], X_train[val_idx]
#             y_tr, y_val = y_train[train_idx], y_train[val_idx]
            
#             model = CatBoostRegressor(**cat_params)
#             model.fit(
#                 X_tr, y_tr,
#                 eval_set=(X_val, y_val),
#                 verbose=0
#             )
            
#             val_pred = model.predict(X_val)
#             cat_oof[val_idx] = val_pred
#             cat_test_preds += model.predict(X_test) / N_FOLDS
            
#             fold_r2 = r2_score(y_val, val_pred)
#             cat_scores.append(fold_r2)
#             print(f"  Fold {fold+1}: R2 = {fold_r2:.6f}")
            
#         overall_r2 = r2_score(y_train, cat_oof)
#         print(f"  CatBoost OOF R2: {overall_r2:.6f}")
#         print(f"  CatBoost Mean CV R2: {np.mean(cat_scores):.6f} +/- {np.std(cat_scores):.6f}")
        
#         results['CatBoost'] = {'mean_r2': np.mean(cat_scores), 'std_r2': np.std(cat_scores), 'oof_r2': overall_r2}
#         oof_predictions['CatBoost'] = cat_oof
#         test_predictions['CatBoost'] = cat_test_preds
        
#     # ---- 2.4 Extra Trees ----
#     print("\n--- Training Extra Trees ---")
#     et_oof = np.zeros(len(X_train))
#     et_test_preds = np.zeros(len(X_test))
#     et_scores = []
    
#     for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
#         X_tr, X_val = X_train[train_idx], X_train[val_idx]
#         y_tr, y_val = y_train[train_idx], y_train[val_idx]
        
#         model = ExtraTreesRegressor(
#             n_estimators=500,
#             max_depth=20,
#             min_samples_split=5,
#             min_samples_leaf=2,
#             max_features='sqrt',
#             random_state=RANDOM_STATE,
#             n_jobs=-1
#         )
#         model.fit(X_tr, y_tr)
        
#         val_pred = model.predict(X_val)
#         et_oof[val_idx] = val_pred
#         et_test_preds += model.predict(X_test) / N_FOLDS
        
#         fold_r2 = r2_score(y_val, val_pred)
#         et_scores.append(fold_r2)
#         print(f"  Fold {fold+1}: R2 = {fold_r2:.6f}")
        
#     overall_r2 = r2_score(y_train, et_oof)
#     print(f"  Extra Trees OOF R2: {overall_r2:.6f}")
#     print(f"  Extra Trees Mean CV R2: {np.mean(et_scores):.6f} +/- {np.std(et_scores):.6f}")
    
#     results['ExtraTrees'] = {'mean_r2': np.mean(et_scores), 'std_r2': np.std(et_scores), 'oof_r2': overall_r2}
#     oof_predictions['ExtraTrees'] = et_oof
#     test_predictions['ExtraTrees'] = et_test_preds
    
#     print("\n" + "=" * 70)
#     print("STEP 3: ENSEMBLE BLENDING SEARCH")
#     print("=" * 70)
    
#     available_models = list(test_predictions.keys())
#     print(f"Available models for ensemble: {available_models}")
    
#     final_test_preds = None
#     final_r2 = -np.inf
#     best_weights_opt = {}
    
#     if len(available_models) > 1:
#         # Search for optimal blend weights via grid search
#         print("\n--- Searching for optimal blend weights ---")
#         best_r2 = -np.inf
#         best_w = None
        
#         if len(available_models) == 2:
#             for w1 in np.arange(0, 1.01, 0.05):
#                 w2 = 1 - w1
#                 blend = w1 * oof_predictions[available_models[0]] + w2 * oof_predictions[available_models[1]]
#                 r2 = r2_score(y_train, blend)
#                 if r2 > best_r2:
#                     best_r2 = r2
#                     best_w = {available_models[0]: w1, available_models[1]: w2}
                    
#         elif len(available_models) == 3:
#             for w1 in np.arange(0, 1.01, 0.05):
#                 for w2 in np.arange(0, 1.01 - w1, 0.05):
#                     w3 = 1 - w1 - w2
#                     blend = (w1 * oof_predictions[available_models[0]] +
#                              w2 * oof_predictions[available_models[1]] +
#                              w3 * oof_predictions[available_models[2]])
#                     r2 = r2_score(y_train, blend)
#                     if r2 > best_r2:
#                         best_r2 = r2
#                         best_w = {available_models[0]: w1, available_models[1]: w2, available_models[2]: w3}
                        
#         elif len(available_models) == 4:
#             for w1 in np.arange(0, 1.01, 0.1):
#                 for w2 in np.arange(0, 1.01 - w1, 0.1):
#                     for w3 in np.arange(0, 1.01 - w1 - w2, 0.1):
#                         w4 = 1 - w1 - w2 - w3
#                         blend = (w1 * oof_predictions[available_models[0]] +
#                                  w2 * oof_predictions[available_models[1]] +
#                                  w3 * oof_predictions[available_models[2]] +
#                                  w4 * oof_predictions[available_models[3]])
#                         r2 = r2_score(y_train, blend)
#                         if r2 > best_r2:
#                             best_r2 = r2
#                             best_w = {
#                                 available_models[0]: w1,
#                                 available_models[1]: w2,
#                                 available_models[2]: w3,
#                                 available_models[3]: w4
#                             }
                            
#         print(f"  Best weight combination: {best_w}")
#         print(f"  Optimal blend OOF R2:    {best_r2:.6f}")
#         print(f"  Optimal blend Score:     {max(0, 100 * best_r2):.2f}")
        
#         optimal_test = sum(best_w[name] * test_predictions[name] for name in available_models)
        
#         # Calculate simple average
#         simple_oof = sum(oof_predictions[name] for name in available_models) / len(available_models)
#         simple_test = sum(test_predictions[name] for name in available_models) / len(available_models)
#         simple_r2 = r2_score(y_train, simple_oof)
#         print(f"\n  Simple Average OOF R2:   {simple_r2:.6f}")
#         print(f"  Simple Average Score:    {max(0, 100 * simple_r2):.2f}")
        
#         # Select best approach
#         if best_r2 >= simple_r2:
#             print("\n*** Selected Ensemble Method: OPTIMAL WEIGHTED BLEND ***")
#             final_r2 = best_r2
#             final_test_preds = optimal_test
#             best_weights_opt = best_w
#         else:
#             print("\n*** Selected Ensemble Method: SIMPLE AVERAGE ***")
#             final_r2 = simple_r2
#             final_test_preds = simple_test
#             best_weights_opt = {name: 1.0 / len(available_models) for name in available_models}
            
#     else:
#         # Only one model trained
#         model_name = available_models[0]
#         final_test_preds = test_predictions[model_name]
#         final_r2 = results[model_name]['oof_r2']
#         best_weights_opt = {model_name: 1.0}
#         print(f"\nUsing single model: {model_name}")
#         print(f"OOF R2: {final_r2:.6f} | Score: {max(0, 100 * final_r2):.2f}")
        
#     # Check if a single model outperforms the ensemble
#     best_single = max(results.items(), key=lambda x: x[1]['oof_r2'])
#     if best_single[1]['oof_r2'] > final_r2:
#         print(f"\n[NOTE] Best single model ({best_single[0]}) beats ensemble!")
#         print(f"  Single R2: {best_single[1]['oof_r2']:.6f} vs Ensemble R2: {final_r2:.6f}")
#         print(f"  Using best single model predictions instead.")
#         final_test_preds = test_predictions[best_single[0]]
#         final_r2 = best_single[1]['oof_r2']
#         best_weights_opt = {name: 1.0 if name == best_single[0] else 0.0 for name in available_models}
        
#     print(f"\nFINAL CV R2 ACHIEVED:  {final_r2:.6f}")
#     print(f"FINAL LEADERBOARD SCORE: {max(0, 100 * final_r2):.2f}")
    
#     print("\n" + "=" * 70)
#     print("STEP 4: CREATING FINAL SUBMISSION")
#     print("=" * 70)
    
#     # Clip predictions to be non-negative
#     final_test_preds = np.clip(final_test_preds, 0, None)
    
#     submission = pd.DataFrame({
#         'Index': test_indices,
#         'demand': final_test_preds
#     })
    
#     # Ensure correct shape
#     print(f"Submission shape: {submission.shape} (Expected: (41778, 2))")
#     assert submission.shape == (41778, 2), f"Incorrect shape! Got {submission.shape}"
    
#     submission_path = os.path.join(OUTPUT_DIR, 'submission.csv')
#     submission.to_csv(submission_path, index=False)
#     print(f"  [SAVED] {submission_path}")
    
#     # Save Results Summary
#     summary = {
#         'models': {name: {k: float(v) for k, v in res.items()} for name, res in results.items()},
#         'ensemble_weights': {k: float(v) for k, v in best_weights_opt.items()},
#         'final_r2': float(final_r2),
#         'final_score': float(max(0, 100 * final_r2)),
#         'n_features': len(feature_cols),
#         'submission_shape': list(submission.shape)
#     }
    
#     summary_path = os.path.join(OUTPUT_DIR, 'results_summary.json')
#     with open(summary_path, 'w') as f:
#         json.dump(summary, f, indent=2)
#     print(f"  [SAVED] {summary_path}")
    
#     print("\nMODEL TRAINING AND ENSEMBLING COMPLETE!")
#     print("=" * 70)

# if __name__ == "__main__":
#     main()



"""
=============================================================================
Model Training & Ensembling - Traffic Demand Prediction (Flipkart GRID)
=============================================================================
Loads pre-extracted features from 'dataset/train_features.csv' and 'dataset/test_features.csv'.
Trains 4 models (LightGBM, XGBoost, CatBoost, ExtraTrees) with 5-Fold Cross-Validation.
Searches for the optimal ensemble weights and outputs:
- Final ensemble submission: 'output/submission.csv'
- Results summary: 'output/results_summary.json'
- Feature importance plot: 'output/lgb_feature_importance.png'
=============================================================================
"""

import pandas as pd
import numpy as np
import os
import json
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Modeling
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.ensemble import ExtraTreesRegressor

warnings.filterwarnings('ignore')

# Boosting libraries with fallbacks
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("[WARN] XGBoost not available. Will skip XGBoost.")

try:
    from lightgbm import LGBMRegressor
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("[WARN] LightGBM not available. Will skip LightGBM.")

try:
    from catboost import CatBoostRegressor
    HAS_CAT = True
except ImportError:
    HAS_CAT = False
    print("[WARN] CatBoost not available. Will skip CatBoost.")

# Configuration
DATA_DIR = "dataset"
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42
N_FOLDS = 5
np.random.seed(RANDOM_STATE)

def main():
    print("=" * 70)
    print("STEP 1: LOADING FEATURE-EXTRACTED DATASETS")
    print("=" * 70)
    
    train_features_path = os.path.join(DATA_DIR, "train_features.csv")
    test_features_path = os.path.join(DATA_DIR, "test_features.csv")
    
    if not os.path.exists(train_features_path) or not os.path.exists(test_features_path):
        raise FileNotFoundError(
            "Feature-extracted CSVs not found! Please run 'extract_features.py' first."
        )
        
    train_df = pd.read_csv(train_features_path)
    test_df = pd.read_csv(test_features_path)
    
    print(f"Loaded Train Features shape: {train_df.shape}")
    print(f"Loaded Test Features shape:  {test_df.shape}")
    
    # Identify target and feature columns
    target_col = 'demand'
    exclude_cols = ['demand', 'Index']
    feature_cols = [c for c in train_df.columns if c not in exclude_cols]
    
    print(f"Total features: {len(feature_cols)}")
    
    # Prepare NumPy arrays for modeling
    X_train = train_df[feature_cols].values.astype(np.float64)
    y_train = train_df[target_col].values.astype(np.float64)
    X_test = test_df[feature_cols].values.astype(np.float64)
    test_indices = test_df['Index'].values.astype(int)
    
    # Replace any remaining NaNs or Infs
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
    
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape:  {X_test.shape}")
    
    print("\n" + "=" * 70)
    print("STEP 2: MODEL TRAINING WITH 5-FOLD CROSS-VALIDATION")
    print("=" * 70)
    
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    results = {}
    oof_predictions = {}
    test_predictions = {}
    
    # ---- 2.1 LightGBM ----
    if HAS_LGB:
        print("\n--- Training LightGBM ---")
        lgb_params = {
            'n_estimators': 2000,
            'learning_rate': 0.03,
            'max_depth': 8,
            'num_leaves': 63,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': RANDOM_STATE,
            'n_jobs': -1,
            'verbose': -1
        }
        
        lgb_oof = np.zeros(len(X_train))
        lgb_test_preds = np.zeros(len(X_test))
        lgb_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]
            
            model = LGBMRegressor(**lgb_params)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                callbacks=[
                    lgb.early_stopping(100, verbose=False),
                    lgb.log_evaluation(0)
                ]
            )
            
            val_pred = model.predict(X_val)
            lgb_oof[val_idx] = val_pred
            lgb_test_preds += model.predict(X_test) / N_FOLDS
            
            fold_r2 = r2_score(y_val, val_pred)
            lgb_scores.append(fold_r2)
            print(f"  Fold {fold+1}: R2 = {fold_r2:.6f}")
            
        overall_r2 = r2_score(y_train, lgb_oof)
        print(f"  LightGBM OOF R2: {overall_r2:.6f}")
        print(f"  LightGBM Mean CV R2: {np.mean(lgb_scores):.6f} +/- {np.std(lgb_scores):.6f}")
        
        results['LightGBM'] = {'mean_r2': np.mean(lgb_scores), 'std_r2': np.std(lgb_scores), 'oof_r2': overall_r2}
        oof_predictions['LightGBM'] = lgb_oof
        test_predictions['LightGBM'] = lgb_test_preds
        
        # Save feature importance plot using LGBM
        fi = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n  Top 10 Features (LightGBM):")
        print(fi.head(10).to_string(index=False))
        
        fig, ax = plt.subplots(figsize=(10, 10))
        fi_top = fi.head(25)
        ax.barh(range(len(fi_top)), fi_top['importance'].values, color='steelblue')
        ax.set_yticks(range(len(fi_top)))
        ax.set_yticklabels(fi_top['feature'].values)
        ax.invert_yaxis()
        ax.set_title('LightGBM Feature Importance (Top 25)', fontsize=14)
        ax.set_xlabel('Importance')
        plt.tight_layout()
        fi_path = os.path.join(OUTPUT_DIR, 'lgb_feature_importance.png')
        plt.savefig(fi_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  [SAVED] {fi_path}")
        
    # ---- 2.2 XGBoost ----
    if HAS_XGB:
        print("\n--- Training XGBoost ---")
        xgb_params = {
            'n_estimators': 2000,
            'learning_rate': 0.03,
            'max_depth': 7,
            'min_child_weight': 5,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'tree_method': 'hist',
            'random_state': RANDOM_STATE,
            'n_jobs': -1,
            'verbosity': 0
        }
        
        xgb_oof = np.zeros(len(X_train))
        xgb_test_preds = np.zeros(len(X_test))
        xgb_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]
            
            model = XGBRegressor(**xgb_params)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            
            val_pred = model.predict(X_val)
            xgb_oof[val_idx] = val_pred
            xgb_test_preds += model.predict(X_test) / N_FOLDS
            
            fold_r2 = r2_score(y_val, val_pred)
            xgb_scores.append(fold_r2)
            print(f"  Fold {fold+1}: R2 = {fold_r2:.6f}")
            
        overall_r2 = r2_score(y_train, xgb_oof)
        print(f"  XGBoost OOF R2: {overall_r2:.6f}")
        print(f"  XGBoost Mean CV R2: {np.mean(xgb_scores):.6f} +/- {np.std(xgb_scores):.6f}")
        
        results['XGBoost'] = {'mean_r2': np.mean(xgb_scores), 'std_r2': np.std(xgb_scores), 'oof_r2': overall_r2}
        oof_predictions['XGBoost'] = xgb_oof
        test_predictions['XGBoost'] = xgb_test_preds
        
    # ---- 2.3 CatBoost ----
    if HAS_CAT:
        print("\n--- Training CatBoost ---")
        cat_params = {
            'iterations': 2000,
            'learning_rate': 0.05,
            'depth': 8,
            'l2_leaf_reg': 3,
            'random_seed': RANDOM_STATE,
            'verbose': 0,
            'early_stopping_rounds': 100,
        }
        
        cat_oof = np.zeros(len(X_train))
        cat_test_preds = np.zeros(len(X_test))
        cat_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]
            
            model = CatBoostRegressor(**cat_params)
            model.fit(
                X_tr, y_tr,
                eval_set=(X_val, y_val),
                verbose=0
            )
            
            val_pred = model.predict(X_val)
            cat_oof[val_idx] = val_pred
            cat_test_preds += model.predict(X_test) / N_FOLDS
            
            fold_r2 = r2_score(y_val, val_pred)
            cat_scores.append(fold_r2)
            print(f"  Fold {fold+1}: R2 = {fold_r2:.6f}")
            
        overall_r2 = r2_score(y_train, cat_oof)
        print(f"  CatBoost OOF R2: {overall_r2:.6f}")
        print(f"  CatBoost Mean CV R2: {np.mean(cat_scores):.6f} +/- {np.std(cat_scores):.6f}")
        
        results['CatBoost'] = {'mean_r2': np.mean(cat_scores), 'std_r2': np.std(cat_scores), 'oof_r2': overall_r2}
        oof_predictions['CatBoost'] = cat_oof
        test_predictions['CatBoost'] = cat_test_preds
        
    # ---- 2.4 Extra Trees ----
    print("\n--- Training Extra Trees ---")
    et_oof = np.zeros(len(X_train))
    et_test_preds = np.zeros(len(X_test))
    et_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        
        model = ExtraTreesRegressor(
            n_estimators=500,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        model.fit(X_tr, y_tr)
        
        val_pred = model.predict(X_val)
        et_oof[val_idx] = val_pred
        et_test_preds += model.predict(X_test) / N_FOLDS
        
        fold_r2 = r2_score(y_val, val_pred)
        et_scores.append(fold_r2)
        print(f"  Fold {fold+1}: R2 = {fold_r2:.6f}")
        
    overall_r2 = r2_score(y_train, et_oof)
    print(f"  Extra Trees OOF R2: {overall_r2:.6f}")
    print(f"  Extra Trees Mean CV R2: {np.mean(et_scores):.6f} +/- {np.std(et_scores):.6f}")
    
    results['ExtraTrees'] = {'mean_r2': np.mean(et_scores), 'std_r2': np.std(et_scores), 'oof_r2': overall_r2}
    oof_predictions['ExtraTrees'] = et_oof
    test_predictions['ExtraTrees'] = et_test_preds
    
    print("\n" + "=" * 70)
    print("STEP 3: ENSEMBLE BLENDING SEARCH")
    print("=" * 70)
    
    available_models = list(test_predictions.keys())
    print(f"Available models for ensemble: {available_models}")
    
    final_test_preds = None
    final_r2 = -np.inf
    best_weights_opt = {}
    
    if len(available_models) > 1:
        # Search for optimal blend weights via grid search
        print("\n--- Searching for optimal blend weights ---")
        best_r2 = -np.inf
        best_w = None
        
        if len(available_models) == 2:
            for w1 in np.arange(0, 1.01, 0.05):
                w2 = 1 - w1
                blend = w1 * oof_predictions[available_models[0]] + w2 * oof_predictions[available_models[1]]
                r2 = r2_score(y_train, blend)
                if r2 > best_r2:
                    best_r2 = r2
                    best_w = {available_models[0]: w1, available_models[1]: w2}
                    
        elif len(available_models) == 3:
            for w1 in np.arange(0, 1.01, 0.05):
                for w2 in np.arange(0, 1.01 - w1, 0.05):
                    w3 = 1 - w1 - w2
                    blend = (w1 * oof_predictions[available_models[0]] +
                             w2 * oof_predictions[available_models[1]] +
                             w3 * oof_predictions[available_models[2]])
                    r2 = r2_score(y_train, blend)
                    if r2 > best_r2:
                        best_r2 = r2
                        best_w = {available_models[0]: w1, available_models[1]: w2, available_models[2]: w3}
                        
        elif len(available_models) == 4:
            for w1 in np.arange(0, 1.01, 0.1):
                for w2 in np.arange(0, 1.01 - w1, 0.1):
                    for w3 in np.arange(0, 1.01 - w1 - w2, 0.1):
                        w4 = 1 - w1 - w2 - w3
                        blend = (w1 * oof_predictions[available_models[0]] +
                                 w2 * oof_predictions[available_models[1]] +
                                 w3 * oof_predictions[available_models[2]] +
                                 w4 * oof_predictions[available_models[3]])
                        r2 = r2_score(y_train, blend)
                        if r2 > best_r2:
                            best_r2 = r2
                            best_w = {
                                available_models[0]: w1,
                                available_models[1]: w2,
                                available_models[2]: w3,
                                available_models[3]: w4
                            }
                            
        print(f"  Best weight combination: {best_w}")
        print(f"  Optimal blend OOF R2:    {best_r2:.6f}")
        print(f"  Optimal blend Score:     {max(0, 100 * best_r2):.2f}")
        
        optimal_test = sum(best_w[name] * test_predictions[name] for name in available_models)
        
        # Calculate simple average
        simple_oof = sum(oof_predictions[name] for name in available_models) / len(available_models)
        simple_test = sum(test_predictions[name] for name in available_models) / len(available_models)
        simple_r2 = r2_score(y_train, simple_oof)
        print(f"\n  Simple Average OOF R2:   {simple_r2:.6f}")
        print(f"  Simple Average Score:    {max(0, 100 * simple_r2):.2f}")
        
        # Select best approach
        if best_r2 >= simple_r2:
            print("\n*** Selected Ensemble Method: OPTIMAL WEIGHTED BLEND ***")
            final_r2 = best_r2
            final_test_preds = optimal_test
            best_weights_opt = best_w
        else:
            print("\n*** Selected Ensemble Method: SIMPLE AVERAGE ***")
            final_r2 = simple_r2
            final_test_preds = simple_test
            best_weights_opt = {name: 1.0 / len(available_models) for name in available_models}
            
    else:
        # Only one model trained
        model_name = available_models[0]
        final_test_preds = test_predictions[model_name]
        final_r2 = results[model_name]['oof_r2']
        best_weights_opt = {model_name: 1.0}
        print(f"\nUsing single model: {model_name}")
        print(f"OOF R2: {final_r2:.6f} | Score: {max(0, 100 * final_r2):.2f}")
        
    # Check if a single model outperforms the ensemble
    best_single = max(results.items(), key=lambda x: x[1]['oof_r2'])
    if best_single[1]['oof_r2'] > final_r2:
        print(f"\n[NOTE] Best single model ({best_single[0]}) beats ensemble!")
        print(f"  Single R2: {best_single[1]['oof_r2']:.6f} vs Ensemble R2: {final_r2:.6f}")
        print(f"  Using best single model predictions instead.")
        final_test_preds = test_predictions[best_single[0]]
        final_r2 = best_single[1]['oof_r2']
        best_weights_opt = {name: 1.0 if name == best_single[0] else 0.0 for name in available_models}
        
    print(f"\nFINAL CV R2 ACHIEVED:  {final_r2:.6f}")
    print(f"FINAL LEADERBOARD SCORE: {max(0, 100 * final_r2):.2f}")
    
    print("\n" + "=" * 70)
    print("STEP 4: CREATING FINAL SUBMISSION")
    print("=" * 70)
    
    # Clip predictions to be non-negative
    final_test_preds = np.clip(final_test_preds, 0, None)
    
    submission = pd.DataFrame({
        'Index': test_indices,
        'demand': final_test_preds
    })
    
    # Ensure correct shape
    print(f"Submission shape: {submission.shape} (Expected: (41778, 2))")
    assert submission.shape == (41778, 2), f"Incorrect shape! Got {submission.shape}"
    
    submission_path = os.path.join(OUTPUT_DIR, 'submission.csv')
    submission.to_csv(submission_path, index=False)
    print(f"  [SAVED] {submission_path}")
    
    # Save Results Summary
    summary = {
        'models': {name: {k: float(v) for k, v in res.items()} for name, res in results.items()},
        'ensemble_weights': {k: float(v) for k, v in best_weights_opt.items()},
        'final_r2': float(final_r2),
        'final_score': float(max(0, 100 * final_r2)),
        'n_features': len(feature_cols),
        'submission_shape': list(submission.shape)
    }
    
    summary_path = os.path.join(OUTPUT_DIR, 'results_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  [SAVED] {summary_path}")
    
    print("\nMODEL TRAINING AND ENSEMBLING COMPLETE!")
    print("=" * 70)

if __name__ == "__main__":
    main()
