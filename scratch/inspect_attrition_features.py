import pandas as pd

def main():
    df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")
    print("Dataset shape:", df.shape)
    print("\nColumns and Dtypes:")
    print(df.dtypes)
    print("\nMissing values:")
    print(df.isnull().sum().sum())
    print("\nCategorical columns value counts:")
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        print(f" -> {col}: {df[col].nunique()} unique values: {df[col].unique()[:5]}")

if __name__ == "__main__":
    main()
