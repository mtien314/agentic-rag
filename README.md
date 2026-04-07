# Description
A project built with Agentic rag using gpt-oss combine local knowledge for accuracy, context-awareness and explainable response

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
uv pip install -r pyproject.toml
```

```
uv run src/main.py
```