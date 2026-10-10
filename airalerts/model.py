"""Logistic regression forecaster, refit at every walk-forward fold.

Same call signature as the baselines: fn(starts, y, train_end, times) -> wide P(y=1).
"""
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

from . import config


def make_logreg(features: pd.DataFrame, train_weeks: int, C: float = 1.0):
    """features: long frame indexed by (t, region), from features.build_features."""
    num_cols = list(features.columns)

    def forecaster(starts, y, train_end, times):
        last_t = train_end - pd.Timedelta(hours=config.HORIZON_H)   # label window closed
        first_t = train_end - pd.Timedelta(weeks=train_weeks)
        y_tr = y.loc[first_t:last_t].stack().rename("y")
        y_tr.index.names = ["t", "region"]
        X_tr = features.join(y_tr, how="inner")
        model = make_pipeline(
            ColumnTransformer([
                ("num", StandardScaler(), num_cols),
                ("region", OneHotEncoder(handle_unknown="ignore"), ["region"]),
            ]),
            LogisticRegression(C=C, max_iter=2000),
        )
        Xf = X_tr.reset_index()
        model.fit(Xf[num_cols + ["region"]], Xf["y"])

        X_te = features.loc[features.index.get_level_values("t").isin(times)].reset_index()
        p = model.predict_proba(X_te[num_cols + ["region"]])[:, 1]
        return (pd.Series(p, index=pd.MultiIndex.from_frame(X_te[["t", "region"]]))
                  .unstack().reindex(index=times, columns=y.columns))

    forecaster.model_name = f"logreg_{train_weeks}w"
    return forecaster
