# Industrial RAG Assistant

An end-to-end Retrieval-Augmented Generation (RAG) application for answering Electrical Engineering questions from a technical PDF document. The project uses semantic chunking, sentence-transformer embeddings, ChromaDB vector search, and a local Llama 3.2 model through Ollama to generate grounded responses.

## Project Overview

Industrial RAG Assistant is a document-based AI assistant that uses Retrieval-Augmented Generation (RAG) to answer Electrical Engineering questions from technical documents. It retrieves relevant information from the source document using semantic search and uses a local Llama 3.2 model to generate grounded answers based only on the retrieved context.


## Features

* PDF document loading and text extraction
* Semantic text chunking with overlap
* Sentence Transformer embeddings
* Persistent ChromaDB vector database
* Similarity-based document retrieval
* Context-grounded prompt generation
* Local LLM inference using Ollama
* Interactive question-answering through the terminal
* Automated evaluation with RAGAS *(planned)*

## Project Structure

```text
Industrial_RAG_Assistant/
│
├── Data/
│   └── ElectricalEngineering.pdf
│
├── Src/
│   ├── pdf_loader.py
│   ├── Chunker.py
│   ├── Embeddings.py
│   ├── VectorDB.py
│   ├── Retriever.py
│   ├── Prompt.py
│   ├── LLM.py
│   └── RAGPipeline.py
│
├── scripts/
│   └── ingest.py
│
├── tests/
│
├── .gitignore
├── main.py
└── README.md
```

## Technologies

* Python
* LangChain Text Splitters
* Sentence Transformers
* ChromaDB
* Ollama
* Llama 3.2
* Pytest
* RAGAS

## RAG Pipeline

```text
PDF Document
     ↓
Text Extraction
     ↓
Text Chunking
     ↓
Embeddings
     ↓
ChromaDB
     ↓
User Question
     ↓
Query Embedding
     ↓
Similarity Retrieval
     ↓
Relevant Context
     ↓
Prompt
     ↓
Llama 3.2
     ↓
Grounded Answer
```

## Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Install and run Ollama, then make sure the Llama model is available:

```powershell
ollama pull llama3.2:3b
```

## Ingest Documents

Run the ingestion script to process the PDF, create embeddings, and store them in ChromaDB:

```powershell
python scripts/ingest.py
```

## Run the Application

Start the RAG assistant:

```powershell
python main.py
```

You can ask multiple questions interactively and type:

```text
exit
```

to stop the application.

## Example

```text
Enter your query: What is a DC motor?

--- Answer ---
The information is not available in the provided context.
```

The assistant is designed to avoid generating unsupported information when the retrieved context does not contain the answer.

## Evaluation

The project will be evaluated using a 25-question evaluation dataset containing questions and reference answers.

RAGAS will be used to evaluate:

* Faithfulness
* Answer Relevancy
* Context Precision
* Context Recall

The evaluation will help identify whether errors originate from document retrieval or answer generation.

## Future Improvements

* RAGAS evaluation pipeline
* Automated evaluation reports
* Improved retrieval strategies
* Metadata-based filtering
* API deployment using FastAPI
* Web-based user interface
* Additional engineering documents
