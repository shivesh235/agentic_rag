# Agentic RAG System

A secure and privacy-focused RAG (Retrieval-Augmented Generation) system that implements reversible data anonymization for database queries and document processing.

## Features

- **Privacy Protection**
  - Complete database content masking
  - Document content masking using Phi-3-mini
  - Reversible encryption for all masked data
  - Secure mapping storage in PostgreSQL

- **Vector Search**
  - FAISS for efficient similarity search
  - Sentence-transformers for embedding generation
  - Masked document storage

- **Agent Workflow**
  - Query masking
  - Context retrieval
  - Response generation using Phi-3-mini
  - Response unmasking

## Tech Stack

- FastAPI for API endpoints
- LangGraph for agent orchestration
- PostgreSQL for relational data and mapping storage
- FAISS for vector similarity search
- Phi-3-mini for PII detection and query understanding

## Setup

1. Create conda environment:
```bash
conda env create -f environment.yml
conda activate rag_env
```

2. Setup PostgreSQL database:
```bash
cd src/db
python setup_postgres.py
```

3. Configure environment variables:
- Copy `.env.example` to `.env`
- Update the variables with your settings

4. Run the application:
```bash
cd src
uvicorn main:app --reload
```

## API Endpoints

- `/query`: Process natural language queries with privacy protection
- `/upload`: Upload and process documents for vector storage

## Project Structure

```
agentic_rag/
├── src/
│   ├── agents/         # Agent implementation
│   ├── privacy/        # Privacy and masking logic
│   ├── db/            # Database operations
│   └── main.py        # FastAPI application
├── environment.yml     # Conda environment
└── README.md
```

## Security Features

- Edge-based masking
- Encryption of sensitive data
- Secure key management
- Audit logging

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## License

MIT License - see LICENSE file for details
