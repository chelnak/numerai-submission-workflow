import numerapi
import logging

from os import getenv
from os.path import join

napi = numerapi.NumerAPI(verbosity='Error')
logger = logging.getLogger(__name__)

def get_current_round_path(path) -> str:
    current_round = napi.get_current_round()
    return join(path, f'numerai_dataset_{current_round}')


def download_dataset(path) -> str:
    """
    An internal function used to download the latest dataset from the Numerai API
    :param path: The path to where the dataset will be saved
    :return: The base path of the latest unzipped training data
    """

    return napi.download_current_dataset(dest_path=path, unzip=True).strip('.zip')


def submit_prediction(model_id, prediction_path):

    public_id = getenv('NUMERAI_PUBLIC_ID')
    secret_key = getenv('NUMERAI_SECRET_KEY')

    _napi = numerapi.NumerAPI(public_id=public_id, secret_key=secret_key, verbosity='INFO')
    submission_id = _napi.upload_predictions(file_path=prediction_path, model_id=model_id)

    logger.info(f'Submission id: {submission_id}')
    return submission_id
