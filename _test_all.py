import sys
import os
import shutil
import logging
import gensim.downloader as api
from gensim.downloader import base_dir

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("test-downloader-api")

logger.info(sys.version.replace("\n", " "))
if os.path.exists(base_dir):
    logger.warning(f"REMOVE {base_dir}")
    shutil.rmtree(base_dir)

info = api.info()

logger.info("Test models")
failed_models = []
for idx, model_name in enumerate(info["models"]):
    logger.info(f'{model_name} ({idx + 1} of {len(info["models"])})')

    try:
        logger.info(api.load(model_name).most_similar("who"))
    except Exception as e:
        logger.exception(e)
        failed_models.append(model_name)

if failed_models:
    logger.critical(f'FAILED MODELS: {", ".join(failed_models)}')

logger.info("#" * 25)
logger.info("Test datasets")
failed_datasets = []
for idx, dataset_name in enumerate(info["corpora"]):
    logger.info(f'{dataset_name} ({idx + 1} of {len(info["corpora"])})')
    try:
        res = sum(1 for _ in api.load(dataset_name))
        if res == 0:
            raise Exception("empty dataset")

    except Exception as e:
        logger.exception(e)
        failed_datasets.append(dataset_name)

if failed_datasets:
    logger.critical(f'FAILED DATASETS: {", ".join(failed_datasets)}')

logger.info("#" * 25)
if len(failed_datasets + failed_models) == 0:
    logger.info("Successful finished without errors!")

else:

    logger.critical("FIX THIS:")
    logger.critical(f'Models: {", ".join(failed_models)}')
    logger.critical(f'Datasets: {", ".join(failed_datasets)}')
logger.info("#" * 25)
