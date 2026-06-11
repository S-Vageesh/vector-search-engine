# Vector Search Engine

A full-stack semantic search engine that enables meaning-based document retrieval using dense vector embeddings, Approximate Nearest Neighbor (ANN) search, and modern web technologies.

Built with FastAPI, React, PostgreSQL, HNSW indexing, and Sentence Transformers.

## Features

* Semantic document search using vector embeddings
* Approximate Nearest Neighbor retrieval with HNSW indexing
* Multi-file document ingestion
* React-based web interface
* FastAPI REST API
* PostgreSQL persistence layer
* pgvector support with local fallback mode
* Automated test suite
* Docker-ready architecture
* Modern frontend dashboard

## Architecture

```text
Documents
    │
    ▼
Sentence Transformers
    │
    ▼
Vector Embeddings
    │
    ▼
HNSW Index
    │
    ▼
Nearest Neighbor Search
    │
    ▼
FastAPI Backend
    │
    ▼
React Frontend
```

## Technology Stack

### Backend

* Python 3.11
* FastAPI
* PostgreSQL
* psycopg
* Sentence Transformers
* hnswlib
* pytest

### Frontend

* React
* Vite
* TypeScript

### Infrastructure

* Docker
* GitHub

## Project Structure

```text
vector-search-engine/
│
├── frontend/
│   ├── src/
│   └── public/
│
├── src/
│   ├── api.py
│   ├── embeddings.py
│   ├── ingestion.py
│   ├── repository.py
│   ├── search.py
│   └── vector_index.py
│
├── tests/
│
├── docs/
│   └── screenshots/
│
├── requirements.txt
└── README.md
```

## Screenshots

### Dashboard

<img width="1853" height="921" alt="dashboard" src="https://github.com/user-attachments/assets/0a2befb9-e7c2-4d76-b3c2-c02e1df577c2" />


### Document Upload

<img width="1855" height="912" alt="upload" src="https://github.com/user-attachments/assets/7f229c73-48fb-4166-80b8-1fae45c937b4" />


### Semantic Search

<img width="1857" height="907" alt="search" src="https://github.com/user-attachments/assets/5bbfb1a9-ca35-4cde-bb16-c8cb4772b30e" />


## How It Works

### 1. Document Ingestion

Text documents are uploaded through the frontend and processed by the ingestion pipeline.

### 2. Embedding Generation

Sentence Transformer models convert documents into dense vector representations.

### 3. Vector Indexing

Embeddings are inserted into an HNSW index for efficient Approximate Nearest Neighbor retrieval.

### 4. Semantic Retrieval

User queries are embedded into vector space and matched against indexed documents using similarity search.

## Example

Document:

```text
Lionel Messi won multiple Ballon d'Or awards.
Argentina won the FIFA World Cup.
```

Query:

```text
soccer
```

Result:

```text
Lionel Messi won multiple Ballon d'Or awards.
Argentina won the FIFA World Cup.
```

Even though the keyword "soccer" does not appear in the document, semantic similarity enables correct retrieval.

## Running Locally

### Clone Repository

```bash
git clone https://github.com/S-Vageesh/vector-search-engine.git
cd vector-search-engine
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Configure Environment

Create a `.env` file:

```env
DATABASE_URL=postgresql://postgres:password@localhost:5432/vector_search
```

### Run Backend

```bash
python -m uvicorn src.api:app --reload
```

Backend:

```text
http://127.0.0.1:8000
```

### Run Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend:

```text
http://localhost:5173
```

## Testing

Run all tests:

```bash
python -m pytest
```

Example:

```text
22 passed
```

## Future Improvements

* Hybrid keyword + semantic retrieval
* Reranking models
* PDF ingestion support
* User authentication
* Distributed indexing
* Vector database benchmarks
* Cloud deployment pipeline

## Key Concepts Demonstrated

* Semantic Search
* Dense Vector Embeddings
* Approximate Nearest Neighbor Search
* HNSW Graph Indexing
* Vector Databases
* Information Retrieval
* FastAPI Development
* React Frontend Engineering
* PostgreSQL Integration
* Full-Stack Software Development

## License

This project is licensed under the MIT License.

## Author

S Vageesh 

Mathematics and Computing Student

Focused on Databases, Information Retrieval, Machine Learning, and Systems Engineering.
