import joblib
import pandas as pd
import numpy as np

def add_engineered_features(df):
    df = df.copy()
    df['TotalSatisfaction'] = df['JobSatisfaction'] + df['EnvironmentSatisfaction'] + df['RelationshipSatisfaction']
    df['YearsSincePromotionRatio'] = df['YearsSinceLastPromotion'] / (df['YearsAtCompany'] + 1)
    df['YearsWithManagerRatio'] = df['YearsWithCurrManager'] / (df['YearsAtCompany'] + 1)
    df['YearsInRoleRatio'] = df['YearsInCurrentRole'] / (df['YearsAtCompany'] + 1)
    df['IncomePerWorkingYear'] = df['MonthlyIncome'] / (df['TotalWorkingYears'] + 1)
    df['IncomePerAge'] = df['MonthlyIncome'] / df['Age']
    df['CompaniesPerWorkingYear'] = df['NumCompaniesWorked'] / (df['TotalWorkingYears'] + 1)
    df['WorklifeOvertimeInteraction'] = df['WorkLifeBalance'] * (1 - df['OverTime'])
    df['DistanceIncomeRatio'] = df['DistanceFromHome'] / (df['MonthlyIncome'] + 1)
    return df

def main():
    print("Loading serialized model from models/attrition_model.pkl...")
    artifact = joblib.load("models/attrition_model.pkl")
    model = artifact["model"]
    features = artifact["features"]
    threshold = artifact["threshold"]
    
    print("\nModel properties:")
    print(f" - Model type: {type(model)}")
    print(f" - Threshold: {threshold}")
    print(f" - Features expected: {len(features)}")
    
    # Test case matching original app.py features
    input_dict = {
        "Age": 30, "BusinessTravel": 2, "DailyRate": 800,
        "Department": 1, "DistanceFromHome": 10, "Education": 3,
        "EducationField": 1, "EnvironmentSatisfaction": 3, "Gender": 1,
        "HourlyRate": 65, "JobInvolvement": 3, "JobLevel": 2,
        "JobRole": 6, "JobSatisfaction": 3, "MaritalStatus": 1,
        "MonthlyIncome": 5000, "MonthlyRate": 14000, "NumCompaniesWorked": 2,
        "OverTime": 0, "PercentSalaryHike": 12, "PerformanceRating": 3,
        "RelationshipSatisfaction": 3, "StockOptionLevel": 1, "TotalWorkingYears": 8,
        "TrainingTimesLastYear": 2, "WorkLifeBalance": 3, "YearsAtCompany": 3,
        "YearsInCurrentRole": 2, "YearsSinceLastPromotion": 1, "YearsWithCurrManager": 3
    }
    
    # Build dataframe
    input_df = pd.DataFrame([input_dict])
    
    # Dynamically add features
    input_df = add_engineered_features(input_df)
    
    # Reindex to match features list
    input_df = input_df.reindex(columns=features, fill_value=0)
    
    # Run prediction
    prob = model.predict_proba(input_df)[0][1]
    prediction = int(prob >= threshold)
    
    print("\nRunning dummy prediction...")
    print(f" - Attrition Probability: {prob:.4f}")
    print(f" - Attrition Verdict (threshold={threshold}): {prediction}")
    print("\nPipeline test: SUCCESS")

if __name__ == "__main__":
    main()
