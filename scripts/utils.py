import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression


# define number of records and features
N_RECORDS = 3000
N_FEATURES = 20
FLOAT_PRECISION = 5

# fix random seed
np.random.seed(0)


def generate_sample(task_type: str, n_classes: int = 3, random_state: int = 42):
    match task_type:
        case "regression":
            X, y = make_regression(
                n_samples=N_RECORDS,
                n_features=N_FEATURES,
                n_informative=N_FEATURES // 2,
                random_state=random_state,
            )
        case "binary_classification":
            X, y = make_classification(
                n_samples=N_RECORDS,
                n_features=N_FEATURES,
                n_informative=N_FEATURES // 2,
                n_classes=2,
                random_state=random_state,
            )
        case "multiclass_classification":
            X, y = make_classification(
                n_samples=N_RECORDS,
                n_features=N_FEATURES,
                n_informative=N_FEATURES // 2,
                n_classes=n_classes,
                random_state=random_state,
            )
        case _:
            raise ValueError(f"Unknown task type: `{task_type}`")

    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(N_FEATURES)])
    df["target"] = y

    nan_columns = [
        str(x)
        for x in np.random.choice([f"feature_{i}" for i in range(N_FEATURES)], replace=False, size=N_FEATURES // 3)
    ]
    for col in nan_columns:
        df.loc[df.sample(frac=0.1).index, col] = np.nan

    return df
