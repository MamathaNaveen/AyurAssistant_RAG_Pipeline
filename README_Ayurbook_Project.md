# 🧠 Generative Search with OpenAI and Chroma – Ayurbook Project

## 📘 Overview

This project demonstrates a **Generative Search and Retrieval-Augmented Generation (RAG)** pipeline using **OpenAI models** and **ChromaDB**.  
The goal is to create an intelligent system that can **understand and answer questions about Ayurvedic knowledge** by leveraging text data from Ayurvedic books.

Unlike traditional keyword-based search, this system uses **semantic embeddings** to retrieve the most relevant context and combines it with **generative AI capabilities** to produce **coherent, human-like responses**.

---

## 🎯 Project Goals

- Develop a **context-aware search assistant** for Ayurvedic literature.  
- Combine **vector-based document retrieval (ChromaDB)** with **OpenAI’s generative models**.  
- Enable **question-answering and exploration** of Ayurvedic texts using natural language.  
- Demonstrate an **end-to-end RAG workflow** suitable for local or hybrid AI setups.

---

## 📚 Data Sources

- **Primary Dataset:** *Ayurvedic text corpus* prepared from books and digital text repositories.  
- **Dataset Format:** Cleaned and structured textual data stored in plain text or CSV form.  
- **Data Preprocessing:**  
  - Recursive document loading and text splitting using LangChain utilities.  
  - Conversion into small, meaningful chunks for efficient vectorization.  
  - Metadata such as book title, section, or author retained where applicable.

---

## 🧠 System Design

The architecture follows a **RAG (Retrieval-Augmented Generation)** pattern:

```
User Query → Embedding → Vector Store Search → Context Retrieval → OpenAI LLM → Generated Answer
```

### 🔹 Components

| Component | Description |
|------------|-------------|
| **Text Loader & Splitter** | Reads Ayurvedic text files, splits them into manageable chunks. |
| **Embedding Generator** | Converts text chunks into numerical vector representations using OpenAI Embeddings. |
| **Chroma Vector Database** | Stores embeddings for fast similarity-based retrieval. |
| **Retriever** | Searches for contextually similar text chunks using vector similarity. |
| **OpenAI Model (LLM)** | Generates natural language answers using both query and retrieved context. |
| **Gradio / Streamlit UI (optional)** | Provides a user interface for interactive Q&A. |

---

## ⚙️ Implementation Details

1. **Environment Setup**
   - Python 3.10+
   - Dependencies: `langchain`, `chromadb`, `openai`, `gradio`, `tiktoken`
   - OpenAI API key configured via environment variable.

2. **Data Preparation**
   - Documents loaded recursively using LangChain loaders.
   - Splitting into 1000-character chunks with overlap for semantic continuity.

3. **Embedding & Storage**
   - Generated embeddings via `OpenAIEmbeddings`.
   - Stored locally in **ChromaDB** with metadata tags for efficient retrieval.

4. **Query Flow**
   - User enters a natural language question.
   - System computes embedding for the query.
   - Retrieves top similar chunks from ChromaDB.
   - Constructs a prompt combining user query + retrieved context.
   - Sends to OpenAI model for generative completion.

5. **Evaluation**
   - Tested with a variety of Ayurvedic topics (e.g., doshas, herbs, treatments).
   - Responses evaluated for factual consistency and contextual relevance.

---

## 🧩 Tools & Libraries

| Library | Purpose |
|----------|----------|
| **LangChain** | Framework for building RAG pipelines. |
| **ChromaDB** | Vector database for storing and querying embeddings. |
| **OpenAI API** | Embedding and generative model provider. |
| **Gradio** | UI for user-friendly Q&A interface. |
| **tiktoken** | Tokenization and text length management. |
| **Python** | Core language for implementation. |

---

## 🚧 Challenges Faced

| Challenge | Solution |
|------------|-----------|
| **Large text size causing memory issues** | Implemented chunk-based text splitting with overlap. |
| **Embedding generation latency** | Batched embedding generation and caching. |
| **Inconsistent context retrieval** | Fine-tuned chunk size and retrieval parameters (`k` values). |
| **Token limit in OpenAI models** | Dynamic prompt truncation based on token count. |
| **Handling domain-specific Ayurvedic terms** | Preprocessed text for transliteration and synonym normalization. |

---

## 🚀 Future Enhancements

- Integrate **local embedding models (Ollama, Hugging Face)** for offline RAG.  
- Expand dataset coverage with **more classical and modern Ayurvedic literature**.  
- Add **evaluation metrics (BLEU, ROUGE, semantic similarity)** for automated assessment.  
- Implement **multimodal retrieval** (e.g., text + images of herbs).  
- Deploy a **web-based chatbot interface** using Gradio or Streamlit.  

---

## 🧩 Folder Structure (Typical)

```
├── data/
│   ├── ayurveda_texts/
│   └── processed/
├── notebooks/
│   └── Demo_3_2_Generative_Search_with_OpenAI_and_Chroma_Ayurbook.ipynb
├── chroma_db/
├── requirements.txt
└── README.md
```

---

## ✍️ Author

**Mamatha K**  
AI/ML | DevOps | MLOps Enthusiast  
*Project: Generative Search on Ayurvedic Texts with OpenAI and ChromaDB*
