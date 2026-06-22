import pandas as pd

def build_interval_dataset(df, features, target):

    interval_features = (
        df.groupby("_time")[features]
          .sum()
    )

    interval_energy = (
        df.groupby("_time")[target]
          .first()
    )

    interval_df = interval_features.join(interval_energy)

    interval_df = interval_df.fillna(0)

    return interval_df