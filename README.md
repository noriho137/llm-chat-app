# README

## Requirements
* Ubuntu 22.04 LTS
* Chainlit 2.5.5
* Chroma 1.0.7
* Langchain 0.3.25
* Transformers 4.51.3

## Environment variables
Create a `.env` under the `src` directory.
For example, set the following.
```bash
# Log level
LOG_LEVEL=INFO

# For text generation model
MODEL_NAME=elyza/ELYZA-japanese-Llama-2-13b-fast-instruct

QUANTIZATION_METHOD=bitsandbytes

# For Vector DB
DATASET_DIR=./dataset
EMBEDDING_MODEL_NAME=intfloat/multilingual-e5-large
DB_PATH=./chroma
CHUNK_SIZE=256
IS_PERSIST=true
COLLECTION_NAME=my_collection
```
`DATASET_DIR`, `DB_PATH` and `COLLECTION_NAME` should be modified to suit your environment.

## How to run

### Preparation
Clone repositry.
```bash
git clone https://github.com/noriho137/llm-chat-app.git
```

```bash
cd llm-chat-app/
```

Create python virtual environment.
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

Create a vector database from the PDF files in `DATASET_DIR`.
```bash
python setup_db.py
```

## Run chainlit
Run the command
```bash
chainlit run app.py
```

Browse `http://localhost:8000`.

For run with different host and port number, as follows.
```bash
chainlit run --host {IP address} --port {Port number} app.py
```
Then, browse `http://{IP address}:{Port number}`.
