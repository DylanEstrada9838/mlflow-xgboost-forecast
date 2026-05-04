import pandas as pd


def create_features(df, lag):
    df = df.copy()
    df["year"] = df.index.year
    df["weekofyear"] = df.index.isocalendar().week.astype(int)
    #df["dayofweek"] = df.index.dayofweek
    df["month"] = df.index.month
    df["quarter"] = df.index.quarter
    #df["dayofyear"] = df.index.dayofyear

    # Create lag features PER store and item
    df[f"sales_lag_{lag}"] = df.groupby(["store", "item"])["sales"].shift(lag)
    
    return df


def get_weekly_df():
    df = pd.read_csv("data/raw/train.csv")
    df["date"] = pd.to_datetime(df["date"])
    # Group by store and item, resample to weekly, and put store/item back as columns
    weekly_df = df.groupby(["store", "item"]).resample("W", on="date")["sales"].sum().reset_index(["store", "item"])
    features_df = create_features(weekly_df, lag=52).dropna()
    return features_df.sort_index()


def split_train_test(
    features_df, train_start_date, train_end_date, test_start_date, test_end_date
):
    train_df = features_df.loc[train_start_date:train_end_date]
    test_df = features_df.loc[test_start_date:test_end_date]
    return train_df, test_df
