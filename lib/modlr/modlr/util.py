
import pandas as pd
import numpy as np
import csv
import pickle
import os
import logging

from . import numerai
from . import metrics

from os.path import join, isfile, exists
from scipy.stats import percentileofscore
from rich.table import Table
from rich.console import Console

from typing import Any

logger = logging.getLogger(__name__)

def color_metric(metric_value, metric_name) -> str:

    VALIDATION_METRIC_INTERVALS = {
        'mean': (0.013, 0.028),
        'sharpe': (0.53, 1.24),
        'std': (0.0303, 0.0168),
        'max_feature_exposure': (0.4, 0.0661),
        'mmc_mean': (-0.008, 0.008),
        'corr_plus_mmc_sharpe': (0.41, 1.34),
        'max_drawdown': (-0.115, -0.015),
        'feature_neutral_mean': (0.006, 0.022)
    }

    low, high = VALIDATION_METRIC_INTERVALS[metric_name]
    pct = percentileofscore(np.linspace(low, high, 100), metric_value)
    if high <= low:
        pct = 100 - pct
    if pct > 95:  # Excellent
        return f'[bold green]{metric_value:.4f}'
    elif pct > 75:  # Good
        return f'[green]{metric_value:.4f}'
    elif pct > 35:  # Fair
        return f'{metric_value:.4f}'
    else:  # Bad
        return f'[red]{metric_value:.4f}'


def read_csv(file_path, column_names=None) -> pd.DataFrame:
    """
    An internal function to efficently load CSV data

    :param file_path: A path like string
    :return: A pandas dataframe containing the loaded data
    """
    
    logger.info(f'Loading {file_path}')

    if column_names is None:
        with open(file_path, 'r') as f:
            column_names = next(csv.reader(f))
    dtypes = {x: np.float32 for x in column_names if x.startswith(
        ('feature', 'target'))}
    return pd.read_csv(file_path, dtype=dtypes, index_col=0)


def get_metrics_table(df: pd.DataFrame, table_title='Metrics') -> tuple:
    """
    Evaluate and display relevant metrics for Numerai

    :param df: A Pandas DataFrame containing the columns "era", "target" and "prediction"
    :return: A tuple of float containing the metrics
    """

    # Calculate metrics
    _metrics = metrics.get_metrics(df)

    table = Table(title=table_title)
    table.add_column('corr')
    table.add_column('std')
    table.add_column('sharpe')
    table.add_column('feature_exposure')
    table.add_column('max_drawdown')
    table.add_column('payout')

    table.add_row(
        color_metric(_metrics['corrs_mean'], 'mean'),
        color_metric(_metrics['std'], 'std'),
        color_metric(_metrics['numerai_sharpe'], 'sharpe'),
        color_metric(_metrics['max_feature_exposure'], 'max_feature_exposure'),
        color_metric(_metrics['max_drawdown'], 'max_drawdown'),
        str(_metrics['payout'])
    )

    console = Console()
    console.print(table)


def neutralize(df: pd.DataFrame, by, proportion=1.0):

    from sklearn.preprocessing import MinMaxScaler
    import scipy

    def _neutralize(df, columns, by, proportion=1.0):
        scores = df[columns]
        exposures = df[by].values
        scores = scores - proportion * \
            exposures.dot(np.linalg.pinv(exposures).dot(scores))
        return scores / scores.std()

    def _normalize(df):
        X = (df.rank(method="first") - 0.5) / len(df)
        return scipy.stats.norm.ppf(X)

    def normalize_and_neutralize(df, columns, by, proportion=1.0):
        # Convert the scores to a normal distribution
        # df[columns] = _normalize(df[columns])
        df[columns] = _neutralize(df, columns, by, proportion)
        return df[columns]

    df['neutralized_preds'] = df.groupby("era").apply(lambda x: normalize_and_neutralize(
        x, ["prediction"], by, proportion))

    scaler = MinMaxScaler()
    df['prediction'] = scaler.fit_transform(
        df[['neutralized_preds']])

    return df


def save_predictions(df, path):
    """
    Save predictions to a csv.
    :param df: Dataframe containing predictions. E.g df['prediction']
    :param path: Path to where submission file will be saved
    :return: None
    """
    logger.info(f'Saving predictions for submission: {path}')
    df.to_csv(
        path_or_buf=path
    )


def get_blob_client(blob):
    """
    Get a configured instance of a blob client
    """

    from azure.identity import DefaultAzureCredential
    from azure.storage.blob import BlobServiceClient

    account_name = os.getenv('STORAGE_ACCOUNT_NAME')
    container_name = 'models'

    try:
        blob_service_client = BlobServiceClient(
            account_url=f'https://{account_name}.blob.core.windows.net/',
            credential=DefaultAzureCredential()
        )
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob)
        return blob_client
    except Exception as e:
        raise e


def upload_model(path):
    """
    Upload model to azure storage account
    """

    try:
        blob_name = os.path.basename(path)
        client = get_blob_client(blob=blob_name)
        logger.info(f'Uploading blob {blob_name}')
        with open(path, "rb") as data:
            client.upload_blob(data, overwrite=True)
    except Exception as e:
        raise e


def download_model(name, dest_path):
    """
    Download model from azure storage account
    """
    from azure.core.exceptions import ResourceNotFoundError

    full_path = join(os.path.abspath(dest_path), name)
    try:

        logger.info(f'Downloading blob {name} to {full_path}')
        client = get_blob_client(blob=name)
        with open(full_path, "wb") as data:
            data.write(client.download_blob().readall())

        logger.info(f'Loading pre-trained model: {full_path}')
        model = pickle.load(open(full_path, 'rb'))
        return model

    except Exception as e:
        raise e