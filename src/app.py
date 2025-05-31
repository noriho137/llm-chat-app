import logging

import chainlit as cl
from environs import env

from text_generator import TextGenerator, make_context
from vector_store import VectorStore

logging.basicConfig(
    format='%(asctime)s [%(levelname)s] %(module)s.%(funcName)s: %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Load environment variables
env.read_env()
log_level = env.log_level('LOG_LEVEL')
model_name = env.str('MODEL_NAME')
quantization_method = env.str('QUANTIZATION_METHOD', None)
embedding_model_name = env.str('EMBEDDING_MODEL_NAME')
db_path = env.str('DB_PATH', './db')
chunk_size = env.int('CHUNK_SIZE', 256)
is_persist = env.bool('IS_PERSIST', False)
collection_name = env.str('COLLECTION_NAME', 'my_collection')

# Set log level
logger.setLevel(log_level)
logging.getLogger('text_generator').setLevel(log_level)
logging.getLogger('vector_store').setLevel(log_level)

# Text generation model
shared_text_generator = None

# Vector DB
shared_vector_store = None


@cl.on_chat_start
async def on_chat_start():
    logger.debug('start')
    global shared_text_generator, shared_vector_store

    if shared_text_generator is None:
        # Initialize text generation model
        shared_text_generator = TextGenerator(model_name_or_path=model_name,
                                              quantization_method=quantization_method)
    if shared_vector_store is None:
        # Load vector DB
        shared_vector_store = VectorStore(embedding_model_name=embedding_model_name,
                                          db_path=db_path,
                                          chunk_size=chunk_size,
                                          is_persist=is_persist)
        shared_vector_store.get_collection(collection_name=collection_name)

    # Save in session
    cl.user_session.set('text_generator', shared_text_generator)
    cl.user_session.set('vector_store', shared_vector_store)

    await cl.Message(content='ようこそ！何か入力してください。').send()
    logger.debug('end')


@cl.on_message
async def on_message(message):
    logger.debug('start')

    # Get from session
    text_generator = cl.user_session.get('text_generator')
    vector_store = cl.user_session.get('vector_store')

    # Retrieve text relevant to the query by vector search
    query = message.content
    results = vector_store.retrieve(query=query, n_results=5)

    # Format the text so that it can be passed to the prompt
    context = make_context(results)

    # Generate an answer
    answer = text_generator.run(query, context)
    await cl.Message(content=answer).send()
    logger.debug('end')
