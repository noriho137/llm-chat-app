import glob
import logging
import os

from environs import env

from vector_store import VectorStore

logging.basicConfig(
    format='%(asctime)s [%(levelname)s] %(module)s.%(funcName)s: %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Load environment variables
env.read_env()
log_level = env.log_level('LOG_LEVEL')
dataset_dir = env.str('DATASET_DIR')
embedding_model_name_or_path = env.str('EMBEDDING_MODEL_NAME_OR_PATH')
db_path = env.str('DB_PATH', './db')
chunk_size = env.int('CHUNK_SIZE', 256)
is_persist = env.bool('IS_PERSIST', False)
collection_name = env.str('COLLECTION_NAME', 'my_collection')

# Set log level
logger.setLevel(log_level)
logging.getLogger('vector_store').setLevel(log_level)


def main():
    logger.debug('start')

    # List up target files
    target_files = glob.glob(os.path.join(dataset_dir, '*'))

    # Initialize client
    vector_store = VectorStore(embedding_model_name_or_path=embedding_model_name_or_path,
                               db_path=db_path,
                               chunk_size=chunk_size,
                               is_persist=is_persist)

    # Create a collection
    vector_store.create_collection(collection_name=collection_name)

    # Add data to the collection
    vector_store.add_collection(target_files=target_files)

    logger.debug('end')
    return


if __name__ == '__main__':
    main()
