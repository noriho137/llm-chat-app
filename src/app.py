import asyncio
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
log_level = env.log_level('LOG_LEVEL', logging.INFO)
model_name_or_path = env.str('MODEL_NAME_OR_PATH')
quantization_method = env.str('QUANTIZATION_METHOD', None)
embedding_model_name_or_path = env.str('EMBEDDING_MODEL_NAME_OR_PATH')
db_path = env.str('DB_PATH', './db')
chunk_size = env.int('CHUNK_SIZE', 256)
is_persist = env.bool('IS_PERSIST', False)
collection_name = env.str('COLLECTION_NAME', 'my_collection')

# Set log level
logger.setLevel(log_level)
logging.getLogger('text_generator').setLevel(log_level)
logging.getLogger('vector_store').setLevel(log_level)


@cl.cache
def load_text_generator(cached_model_name_or_path, cached_quantization_method):
    text_generator = TextGenerator(model_name_or_path=cached_model_name_or_path,
                                   quantization_method=cached_quantization_method)
    return text_generator


@cl.cache
def load_vector_store(cached_embedding_model_name_or_path, cached_db_path, cached_chunk_size, cached_is_persist, cached_collection_name):
    vector_store = VectorStore(embedding_model_name_or_path=cached_embedding_model_name_or_path,
                               db_path=cached_db_path,
                               chunk_size=cached_chunk_size,
                               is_persist=cached_is_persist)
    vector_store.get_collection(collection_name=cached_collection_name)
    return vector_store


@cl.on_chat_start
async def on_chat_start():
    logger.debug('start')

    message = cl.Message(content='モデルをロード中です。しばらくお待ちください。')
    await message.send()

    with cl.Step(name='モデルをロード', type='llm'):
        text_generator = await asyncio.to_thread(load_text_generator,
                                                 model_name_or_path,
                                                 quantization_method)
        vector_store = await asyncio.to_thread(load_vector_store,
                                               embedding_model_name_or_path,
                                               db_path,
                                               chunk_size,
                                               is_persist,
                                               collection_name)

    # Save in session
    cl.user_session.set('text_generator', text_generator)
    cl.user_session.set('vector_store', vector_store)

    message.content = 'モデルのロードが完了しました。'
    await message.update()

    await cl.Message(content='ようこそ！ご用件は何でしょうか？').send()

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
