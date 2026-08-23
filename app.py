import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import tempfile
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import plotly.express as px

st.set_page_config(page_title="RAG Chatbot + Vector Visualizer", layout="wide")
st.title("📄 Local RAG + 🌐 3D Vector Space Visualizer")

uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    with st.spinner("Processing & Vectorizing document..."):
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        splits = text_splitter.split_documents(docs)

        embeddings_model = OllamaEmbeddings(model="nomic-embed-text")
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings_model)
        retriever = vectorstore.as_retriever()

        llm = ChatOllama(model="llama3.2")
        prompt = ChatPromptTemplate.from_template("""
        Answer the question based strictly on the provided context:
        <context>
        {context}
        </context>
        Question: {question}
        """)

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

    st.success(f"Indexed {len(splits)} chunks into ChromaDB.")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("💬 Ask Your Document")
        user_query = st.text_input("Your Question:")
        if user_query:
            with st.spinner("Thinking..."):
                response = rag_chain.invoke(user_query)
                st.write(response)

    with col2:
        st.subheader("🌌 3D Embedding Space")
        # Extract raw embeddings from Chroma
        raw_data = vectorstore.get(include=["embeddings", "documents"])
        raw_vectors = np.array(raw_data["embeddings"])
        raw_texts = raw_data["documents"]

        if len(raw_vectors) >= 3:
            # Reduce 768-dimensional vectors to 3D coordinates using PCA
            pca = PCA(n_components=3)
            coords = pca.fit_transform(raw_vectors)

            # Format hover preview text
            hover_snippets = [t[:120].replace("\n", " ") + "..." for t in raw_texts]

            df = pd.DataFrame(coords, columns=["PCA_X", "PCA_Y", "PCA_Z"])
            df["Snippet"] = hover_snippets
            df["Chunk_ID"] = [f"Chunk {i+1}" for i in range(len(raw_texts))]

            fig = px.scatter_3d(
                df,
                x="PCA_X",
                y="PCA_Y",
                z="PCA_Z",
                hover_name="Chunk_ID",
                hover_data={"Snippet": True, "PCA_X": False, "PCA_Y": False, "PCA_Z": False},
                title="Semantic Cluster Map",
                color="PCA_X",
                color_continuous_scale="Viridis",
            )
            fig.update_traces(marker=dict(size=6, opacity=0.85))
            fig.update_layout(margin=dict(l=0, r=0, b=0, t=30), height=500)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Upload a document with at least 3 chunks to render 3D vectors.")