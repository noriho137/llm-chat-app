# README

## Requirements
* Ubuntu 22.04 LTS
* Chainlit 2.5.5
* Chroma 1.0.7
* Langchain 0.3.25
* Transformers 4.51.3


## How to run

### Preparation
Clone the repository.
```bash
git clone https://github.com/noriho137/llm-chat-app.git
```

```bash
cd llm-chat-app/
```

Create a python virtual environment.
```bash
python -m venv venv
```

Activate the virtual environment.
```bash
source venv/bin/activate
```

Install python packages.
```bash
pip install -r requirements.txt
```

Create a `.env` file under the `src` directory, based on the `.env.example` file.
```bash
# Log level
LOG_LEVEL=INFO

# For text generation model
MODEL_NAME_OR_PATH=elyza/ELYZA-japanese-Llama-2-13b-fast-instruct
MESSAGE_CONFIG_PATH=config/elyza/ELYZA-japanese-Llama-2-13b-fast-instruct.json
QUANTIZATION_METHOD=bitsandbytes

# For Vector DB
DATASET_DIR=./data
EMBEDDING_MODEL_NAME_OR_PATH=intfloat/multilingual-e5-large
DB_PATH=./chroma
CHUNK_SIZE=256
IS_PERSIST=true
COLLECTION_NAME=my_collection
RETRIEVE_TOP_K=5
```
`DATASET_DIR`, `DB_PATH` and `COLLECTION_NAME` should be modified to suit your environment.

Create a vector database from the PDF files in `DATASET_DIR`.
```bash
cd src/
python setup_db.py
```

### Run chainlit
Run the command
```bash
chainlit run app.py
```

Browse `http://localhost:8000`.

For run with different host and port number, you should add options as follows.
```bash
chainlit run --host {IP address} --port {Port number} app.py
```
Then, browse `http://{IP address}:{Port number}`.
