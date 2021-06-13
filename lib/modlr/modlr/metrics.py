import numpy as np


def correlation(predictions, targets):
    ranked_preds = predictions.rank(pct=True, method="first")
    return np.corrcoef(ranked_preds, targets)[0, 1]


def score(df):
    return correlation(df['prediction'], df['target'])


def get_metrics(df):

    feature_names = [f for f in df.columns if f.startswith('feature')]

    # Calculate metrics
    corrs = df.groupby("era").apply(score)
    corrs_mean = corrs.mean()
    std = corrs.std(ddof=0)
    payout = round((corrs / 0.2).clip(-1, 1).mean(), 4)
    numerai_sharpe = corrs_mean / std

    max_per_era = df.groupby('era').apply(
        lambda d: d[feature_names].corrwith(d['prediction']).abs().max()
    )

    max_feature_exposure = max_per_era.mean()

    rolling_max = (corrs + 1).cumprod().rolling(window=100, min_periods=1).max()
    daily_value = (corrs + 1).cumprod()
    max_drawdown = -((rolling_max - daily_value) / rolling_max).max()

    return {
        'corrs_mean': corrs_mean,
        'std': std,
        'numerai_sharpe': numerai_sharpe,
        'max_feature_exposure': max_feature_exposure,
        'max_drawdown': max_drawdown,
        'payout': payout
    }
