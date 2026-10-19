# Local BI-Assistant

A small, fully local RAG (Retrieval-Augmented Generation) prototype that answers natural-language questions about internal company documents. Everything runs on your own machine: no API keys and no data sent to external services once the models are downloaded.

The sample document is a Portuguese internal policy on holidays, hybrid work and training budget ([documentos/dados_empresa.txt](documentos/dados_empresa.txt)), so the prompts and questions are in Portuguese.

## How it works

[app_rag.py](app_rag.py) runs a five-step pipeline:

1. **Load** the text file in `documentos/dados_empresa.txt`.
2. **Split** it into chunks of 300 characters with a 50-character overlap.
3. **Embed** the chunks with `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` and store them in an in-memory Chroma vector store.
4. **Load** the local LLM `TinyLlama/TinyLlama-1.1B-Chat-v1.0` through a Hugging Face `transformers` pipeline.
5. **Answer**: retrieve the 2 most relevant chunks for the question, place them in a prompt that tells the model to answer only from that context, and print the answer with the source chunks.

```
question ──► retriever (Chroma, k=2) ──► prompt + context ──► TinyLlama ──► answer + sources
```

## Requirements

- Python 3.10 or newer
- About 3 GB of free disk space for the models, downloaded from Hugging Face on the first run
- An internet connection for the first run only; later runs use the local model cache

A GPU is not required. TinyLlama runs on CPU, but generation takes noticeably longer.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Run the script from the project root, because the document path is relative:

```bash
python app_rag.py
```

Example output (the generated answer will vary):

```
1. A carregar o documento da empresa...
2. A gerar Embeddings e a povoar a Vector Database local...
3. A carregar o LLM local (TinyLlama)...
4. A inicializar o sistema de pergunta/resposta RAG...

--- Pergunta: Quando posso marcar minhas férias? ---

--- Resposta Gerada ---
...

--- Fontes Utilizadas ---
- 1. Férias Anuais: Todos os colaboradores têm direito a 22 dias úteis de férias por ano. ...
```

## Customizing

All settings are in [app_rag.py](app_rag.py):

| What | Where |
| --- | --- |
| The question asked | `pergunta` near the end of the file |
| The source document | `document_path` |
| Chunk size and overlap | `RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)` |
| Number of retrieved chunks | `search_kwargs={"k": 2}` |
| Embedding model | `model_name` in `HuggingFaceEmbeddings` |
| LLM | `model_id` |
| Answer length and sampling | `max_new_tokens`, `temperature`, `repetition_penalty` in `pipeline(...)` |
| Prompt instructions | `PromptTemplate.from_template(...)` |

To use your own data, replace the contents of `documentos/dados_empresa.txt` or point `document_path` to another UTF-8 text file.

## Limitations

- Only one plain-text file is loaded; PDFs, spreadsheets and multiple files are not supported yet.
- The vector store lives in memory, so embeddings are rebuilt on every run.
- The question is hard-coded; there is no interactive prompt or UI.
- TinyLlama is a 1.1B-parameter model. It is fast and light but can give imprecise answers, so check them against the printed sources.

## Project structure

```
.
├── app_rag.py                  # RAG pipeline: load, split, embed, retrieve, generate
├── documentos/
│   └── dados_empresa.txt       # Sample company policy document
└── requirements.txt            # Python dependencies
```
