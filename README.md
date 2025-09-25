```
```
# Data Retrieval & Analysis System

Implementation and adaptation by **Aymen Echchalim (2025)**  
Based on original open-source foundations (see LICENSE).  

---

## 📌 Overview
This project implements a robust pipeline for document processing, metadata extraction, and retrieval evaluation. It integrates parsing, storage, and evaluation layers into a cohesive system, with extensible modules for future improvements.

---

## 👨‍💻 My Contributions
- **Parser system for metadata extraction**: Designed and implemented a parsing layer capable of extracting structured metadata from semi-structured/unstructured documents.  
- **Extended functionality for `vector_store`**: Enhanced the vector storage system to support efficient indexing, retrieval, and persistence.  
- **Integrated evaluation protocol**: Developed an evaluation pipeline to benchmark retrieval accuracy and reliability across different configurations.  
- **Documentation & Unit Tests**: Wrote developer-friendly documentation, usage examples, and comprehensive unit tests for all major components.  

---

## 🚀 How to Run

### 1. Clone the repository and set up environment
```bash
git clone <your-repo-url>
cd <your-repo>
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
### 2. Configure environment variables

- Copy `example.env` to `.env`:

```bash
cp example.env .env
```
- Open `.env` and add your **OpenAI API key**.
- Database credentials are already set for the local Docker instance.

### 3. Start PostgreSQL with Timescale & pgvectorscale

- Ensure Docker is running, then launch the container:

```bash
docker compose up -d
```
- The database will be available at:
  - Host: `localhost`
  - Port: `5432`
  - User: `postgres`
  - Password: `password`
  - Database: `postgres`

### 4. Insert data into the vector store

Run the parser and embedding script to extract metadata and insert embeddings into PostgreSQL:

```bash
python insert_vectors.py
```
### 5. Run similarity search queries

Use the query script to test hybrid/semantic retrieval:

```bash
python similarity_search.py "your query here"
```
### 6. Evaluate retrieval performance

Execute the evaluation protocol to benchmark precision, recall, and ranking metrics:

```bash
python evaluate.py --config config/eval.yml
```
### 7. Run unit tests

```bash
pytest tests/
```
---

## 🗂 Project Structure

```
project/
│── parser/              # Metadata extraction system
│── vector_store/        # Extended vector database features
│── evaluation/          # Evaluation protocol
│── docs/                # Documentation
│── tests/               # Unit tests
│── config/              # Config files
│── main.py              # Entry point
│── requirements.txt
│── LICENSE
```
---

## 📝 Notes

- This project builds on existing open-source work but has been extensively adapted and extended.
- For licensing terms of the original code, please refer to the `LICENSE` file.

```

```
