import logging

import chromadb
from chromadb.config import Settings
from langchain_community.document_loaders import PDFMinerLoader
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)


def extract_text(target_file):
    """
    Load a PDF files and extract the text of each page.
    """
    logger.debug('start')

    loader = PDFMinerLoader(target_file)
    text = loader.load()
    pages = text[0].page_content.split('\x0c')

    logger.debug('end')
    return pages


class VectorStore:
    def __init__(self, embedding_model_name_or_path, db_path='./db', chunk_size=256, is_persist=True):
        """
        Initialize client.
        """
        logger.debug('start')

        self.collection = None
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name_or_path)
        self.db_path = db_path
        self.chunk_size = chunk_size
        self.is_persist = is_persist

        # Initialize client
        if self.is_persist:
            # For persist DB
            self.client = chromadb.PersistentClient(path=self.db_path)
        else:
            # For in-memory
            settings = Settings(allow_reset=True)
            self.client = chromadb.EphemeralClient(settings=settings)
            self.client.reset()

        logger.debug('end')
        return

    def create_collection(self, collection_name):
        """
        Create a collection.
        """
        logger.debug('start')
        self.collection = self.client.create_collection(name=collection_name)
        logger.debug('end')
        return

    def get_collection(self, collection_name):
        """
        Get the collection.
        """
        logger.debug('start')
        self.collection = self.client.get_collection(name=collection_name)
        logger.debug('end')
        return

    def add_collection(self, target_files):
        """
        Add data to the collection.
        """
        logger.debug('start')

        for i, target_file in enumerate(target_files):
            pages = extract_text(target_file)
            logger.debug(f'{target_file}: # of pages={len(pages)}')

            for j, page in enumerate(pages):
                text = page.replace('\n', '')
                if text == '':
                    continue

                # Split text by chunk size
                chunks = [page[idx:(idx + self.chunk_size)].replace('\n', '')
                          for idx in range(0, len(page), self.chunk_size)]

                # Convert the text into embedding vectors chunk by chunk
                # and add them to the vector DB.
                for k, chunk in enumerate(chunks):
                    embedded_docs = self.embeddings.embed_documents([chunk])
                    self.collection.add(
                        embeddings=embedded_docs,
                        documents=[chunk],
                        metadatas=[{'source': target_file, 'page': j, 'chunk': k}],
                        ids=[f'F{i:03}-P{j:03}-C{k:03}']
                    )

        logger.debug(f'# of entries={self.collection.count()}')
        logger.debug('end')
        return

    def retrieve(self, query, n_results=5):
        """
        Vector search
        """
        logger.debug('start')

        embedded_query = self.embeddings.embed_query(query)
        results = self.collection.query(
            query_embeddings=embedded_query,
            n_results=n_results,
        )

        logger.debug('end')
        return results
