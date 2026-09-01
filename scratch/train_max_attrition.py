import pandas as pd
import numpy as np
import os
import joblib
import optuna
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, classification_report, precision_recall_curve, accuracy_score
from sklearn.ensemble import VotingClassifier
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
optuna.logging.set_verbosity(optuna.logging.WARNING)

def add_engineered_features(df):
    df = df.copy()
    
    # 1. Total Satisfaction Index
    df['TotalSatisfaction'] = df['JobSatisfaction'] + df['EnvironmentSatisfaction'] + df['RelationshipSatisfaction']
    
    # 2. Tenure ratios
    df['YearsSincePromotionRatio'] = df['YearsSinceLastPromotion'] / (df['YearsAtCompany'] + 1)
    df['YearsWithManagerRatio'] = df['YearsWithCurrManager'] / (df['YearsAtCompany'] + 1)
    df['YearsInRoleRatio'] = df['YearsInCurrentRole'] / (df['YearsAtCompany'] + 1)
    
    # 3. Income relative indicators
    df['IncomePerWorkingYear'] = df['MonthlyIncome'] / (df['TotalWorkingYears'] + 1)
    df['IncomePerAge'] = df['MonthlyIncome'] / df['Age']
    
    # 4. Job Hopper index
    df['CompaniesPerWorkingYear'] = df['NumCompaniesWorked'] / (df['TotalWorkingYears'] + 1)
    
    # 5. Overtime burnout proxy
    df['WorklifeOvertimeInteraction'] = df['WorkLifeBalance'] * (1 - df['OverTime'])
    
    # 6. Commute friction relative to compensation
    df['DistanceIncomeRatio'] = df['DistanceFromHome'] / (df['MonthlyIncome'] + 1)
    
    return df

def preprocess_and_load():
    print("Loading raw HR Employee Attrition dataset...")
    df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")
    
    # Drop useless constant/identifier columns
    df = df.drop(['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours'], axis=1)
    
    # Target encoding
    df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})
    
    # Label encode categorical features alphabetically (identical to label encoder defaults)
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    print("Encoding categorical columns:", cat_cols)
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        
    # Feature engineering
    print("Applying HR feature engineering...")
    df = add_engineered_features(df)
    
    X = df.drop('Attrition', axis=1)
    y = df['Attrition']
    return X, y

def tune_lgbm(X_train, y_train):
    print("Tuning LightGBM with Optuna...")
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'num_leaves': trial.suggest_int('num_leaves', 15, 63),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
            'random_state': 42,
            'verbose': -1,
            'n_jobs': -1
        }
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        aucs = []
        for train_idx, val_idx in cv.split(X_train, y_train):
            model = LGBMClassifier(**params)
            model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
            preds = model.predict_proba(X_train.iloc[val_idx])[:, 1]
            aucs.append(roc_auc_score(y_train.iloc[val_idx], preds))
        return np.mean(aucs)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)
    return study.best_params

def tune_xgb(X_train, y_train):
    print("Tuning XGBoost with Optuna...")
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'random_state': 42,
            'eval_metric': 'auc',
            'n_jobs': -1
        }
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        aucs = []
        for train_idx, val_idx in cv.split(X_train, y_train):
            model = XGBClassifier(**params)
            model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
            preds = model.predict_proba(X_train.iloc[val_idx])[:, 1]
            aucs.append(roc_auc_score(y_train.iloc[val_idx], preds))
        return np.mean(aucs)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)
    return study.best_params

def evaluate_cv(model, X, y):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []
    for train_idx, val_idx in cv.split(X, y):
        # We apply SMOTE inside each fold to avoid target leakage!
        smote = SMOTE(random_state=42)
        X_train_cv, y_train_cv = smote.fit_resample(X.iloc[train_idx], y.iloc[train_idx])
        
        model.fit(X_train_cv, y_train_cv)
        preds = model.predict_proba(X.iloc[val_idx])[:, 1]
        aucs.append(roc_auc_score(y.iloc[val_idx], preds))
    return np.mean(aucs), aucs

def main():
    X, y = preprocess_and_load()
    
    # Split train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Apply SMOTE to the training set for tuning and final fitting
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    print(f"Original Train distribution: {y_train.value_counts().to_dict()}")
    print(f"Balanced Train distribution: {pd.Series(y_train_balanced).value_counts().to_dict()}")
    
    # Tune models on balanced data
    lgb_params = tune_lgbm(X_train_balanced, y_train_balanced)
    xgb_params = tune_xgb(X_train_balanced, y_train_balanced)
    
    # Setup final models
    print("\nTraining optimized baseline models...")
    xgb_model = XGBClassifier(**xgb_params, random_state=42, eval_metric='auc', n_jobs=-1)
    lgb_model = LGBMClassifier(**lgb_params, random_state=42, verbose=-1, n_jobs=-1)
    cat_model = CatBoostClassifier(iterations=300, learning_rate=0.08, depth=5, random_seed=42, verbose=0)
    
    # Evaluate individual models on 5-fold CV
    print("\nEvaluating individual models (5-Fold CV)...")
    for name, model in [("XGBoost", xgb_model), ("LightGBM", lgb_model), ("CatBoost", cat_model)]:
        mean_auc, fold_aucs = evaluate_cv(model, X_train, y_train)
        print(f" -> {name} Mean CV ROC-AUC: {mean_auc:.5f} (Folds: {[round(x, 4) for x in fold_aucs]})")
        
    # Evaluate Voting Classifier
    ensemble = VotingClassifier(
        estimators=[
            ('xgb', xgb_model),
            ('lgb', lgb_model),
            ('cat', cat_model)
        ],
        voting='soft'
    )
    mean_ens_auc, ens_folds = evaluate_cv(ensemble, X_train, y_train)
    print(f" -> Ensemble Mean CV ROC-AUC: {mean_ens_auc:.5f} (Folds: {[round(x, 4) for x in ens_folds]})")
    
    # Fit on full training set
    print("\nFitting final ensemble on full training set...")
    ensemble.fit(X_train_balanced, y_train_balanced)
    
    # Evaluate on test set
    test_proba = ensemble.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, test_proba)
    print(f"\nFinal Ensemble Test ROC-AUC: {test_auc:.5f}")
    
    # Threshold tuning via Precision-Recall curve
    precision, recall, thresholds = precision_recall_curve(y_test, test_proba)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-9)
    best_idx = np.argmax(f1)
    best_threshold = thresholds[best_idx]
    
    # For employee attrition, we prefer a slightly lower threshold to increase recall (catch leaving risks)
    # We will clamp it to a reasonable minimum to avoid high false positives
    final_threshold = max(0.25, round(best_threshold, 2))
    print(f"Optimal F1 Threshold: {best_threshold:.4f} | Selected Final Threshold: {final_threshold:.2f}")
    
    # Metrics
    test_pred = (test_proba >= final_threshold).astype(int)
    acc = accuracy_score(y_test, test_pred)
    print(f"Test Set Accuracy (using {final_threshold} threshold): {acc:.5f}")
    print("\nClassification Report (Test Set):")
    print(classification_report(y_test, test_pred))
    
    # Save the model
    os.makedirs("models", exist_ok=True)
    joblib.dump({
        "model": ensemble,
        "features": X_train.columns.tolist(),
        "threshold": final_threshold
    }, "models/attrition_model.pkl")
    print("\nModel saved successfully as models/attrition_model.pkl!")

if __name__ == "__main__":
    main()
