# Local RAG Document Chatbot & 3D Vector Space Visualizer

A 100% free, fully local Retrieval-Augmented Generation (RAG) web application that allows users to upload PDF documents, query them without cloud APIs, and inspect the 768-dimensional embedding space through an interactive 3D scatter plot.

## Features
- **Zero API Cost & Air-Gapped:** Uses local Llama 3.2 and Nomic-Embed-Text models via Ollama.
- **Sub-second Vector Search:** Local vector storage and similarity retrieval powered by ChromaDB.
- **3D Semantic Visualization:** Employs Principal Component Analysis (PCA) to compress 768D embeddings into a 3D coordinate system visualized with Plotly.
- **Interactive UI:** Clean web interface built with Streamlit.

## Tech Stack
- **Frontend:** Streamlit, Plotly
- **Orchestration:** LangChain (LCEL)
- **Vector Database:** ChromaDB
- **Local Models:** Ollama (Llama 3.2, nomic-embed-text)
- **Data & Math:** scikit-learn (PCA), Pandas, NumPy

## How to Run Locally

1. **Install Ollama & pull the models:**
   ```bash
   pip install -r requirements.txt
   streamlit run app.py
   ollama run llama3.2
   ollama pull nomic-embed-text
   
