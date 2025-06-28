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
top_k = env.int('RETRIEVE_TOP_K', 5)


# Set log level
logger.setLevel(log_level)
logging.getLogger('text_generator').setLevel(log_level)
logging.getLogger('vector_store').setLevel(log_level)


@cl.cache
def load_text_generator(model_name_or_path, quantization_method):
    text_generator = TextGenerator(model_name_or_path=model_name_or_path,
                                   quantization_method=quantization_method)
    return text_generator


@cl.cache
def load_vector_store(embedding_model_name_or_path, db_path, chunk_size, is_persist, collection_name):
    vector_store = VectorStore(embedding_model_name_or_path=embedding_model_name_or_path,
                               db_path=db_path,
                               chunk_size=chunk_size,
                               is_persist=is_persist)
    vector_store.get_collection(collection_name=collection_name)
    return vector_store


@cl.on_chat_start
async def on_chat_start():
    logger.debug('start')

    message = cl.Message(content='モデルをロード中です。しばらくお待ちください。')
    await message.send()

    try:
        with cl.Step(name='モデルをロード', type='llm'):
            # Load text generation model
            text_generator = await asyncio.to_thread(load_text_generator,
                                                     model_name_or_path,
                                                     quantization_method)

            # Load vector DB
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
    except Exception as e:
        logger.exception('Exception occurs in loading model.')
        message.content = f'モデルのロード中にエラーが発生しました。\n{e}'
        await message.update()
        await cl.Message(content='').send()

    logger.debug('end')


@cl.on_message
async def on_message(message):
    logger.debug('start')

    # Get from session
    text_generator = cl.user_session.get('text_generator')
    vector_store = cl.user_session.get('vector_store')

    if text_generator is None:
        logger.error('text_generator is not in the session.')
        await cl.Message(content='システムエラーが発生しました。').send()
        return

    if vector_store is None:
        logger.error('vector_store is not in the session.')
        await cl.Message(content='システムエラーが発生しました。').send()
        return

    try:
        query = message.content

        with cl.Step(name='回答を生成中', type='llm'):
            # Retrieve text relevant to the query by vector search
            results = await asyncio.to_thread(vector_store.retrieve,
                                              query=query,
                                              top_k=top_k)

            # Format the text so that it can be passed to the prompt
            context = make_context(results)

            # Generate an answer
            answer = await asyncio.to_thread(text_generator.run,
                                             query=query,
                                             context=context)

        await cl.Message(content=answer).send()
    except Exception as e:
        logger.exception('Exception occurs in generating answer for query: {message.content}')
        await cl.Message(content=f'回答の生成中にエラーが発生しました。\n{e}').send()

    logger.debug('end')
