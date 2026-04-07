# Description
A project built with Agentic rag using gpt-oss combine local knowledge for accuracy, context-awareness and explainable response about legal VietNam

# Dataset
Dataset: `another-symato/VMTEB-Zalo-legel-retrieval-wseg` . Get corpus 1000
# Structure 
```
┣ 📂src
┃ ┣ 📜main.py
┃ ┣ 📜nodes.py
┃ ┗ 📜tools.py
┣ 📜.env
┣ 📜pyproject.toml
┗ 📜uv.lock
```
```
Vector db: Qdrant
Embedding model: Qwen/Qwen3-Embedding-0.6B
```

# Installation

Create `GROQ_API_KEY`
```
pip install uv
```
```
uv pip install -r pyproject.toml
```

```
uv run src/main.py
```