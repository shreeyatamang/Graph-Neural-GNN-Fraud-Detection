import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_data(path):
    df = pd.read_csv(path)
    return df


def clean_data(df):
    print("Original Shape:", df.shape)

   
    df = df.drop(columns=[
        'oldbalanceOrg',
        'newbalanceOrig',
        'oldbalanceDest',
        'newbalanceDest'
    ])

    
    df = df.drop_duplicates()

    
    df = pd.get_dummies(df, columns=['type'])

   
    scaler = StandardScaler()
    df['amount'] = scaler.fit_transform(df[['amount']])

    print("Cleaned Shape:", df.shape)

    return df


if __name__ == "__main__":
    df = load_data("data/paysim.csv")
    df_clean = clean_data(df)

   
    df_clean.to_csv("data/cleaned_paysim.csv", index=False)

    print(" Data cleaning completed and saved!")