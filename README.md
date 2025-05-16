# Smart Sales Assistant for Wildberries  
**Course**: DS 235 - Generative AI  
**Group Members**: Astghik Grigoryan, Ararat Kazarian  

---

## 📌 Project Overview

The **Smart Sales Assistant for Wildberries** is a generative AI tool designed to analyze mock sales data and generate human-readable summaries from key performance metrics. It enables quick insights for sales performance and decision-making in a scalable and interactive format.

## 🧠 Model Overview

This project uses a two-stage Retrieval-Augmented Generation (RAG) pipeline combining embedding-based retrieval and both generative and extractive answering:

### 🔹 Embedding Model – `all-MiniLM-L6-v2`

**Summary**:  
This model converts product summaries and user queries into vector embeddings, enabling semantic similarity matching even when exact words differ.

**Justification**:  
- Lightweight and fast, ideal for CPU environments.
- Strong performance on semantic textual similarity benchmarks.
- Enables scalable retrieval using vector databases like FAISS.

### 🔹 Retrieval Engine – FAISS

**Summary**:  
Performs similarity search over embedded summaries to find the most relevant content given a user query.

**Justification**:  
- Extremely fast and memory-efficient.
- Scalable for real-time applications.
- Seamlessly integrates with vector representations from transformer models.

### 🔹 Generative QA Model – `google/flan-t5-base`

**Summary**:  
Generates natural-language answers using the retrieved summaries and the user question. A custom prompt is built by combining context with the query.

**Justification**:  
- Fine-tuned for instruction-following and QA tasks.
- Excels at zero-shot generation, producing fluent and informative responses.
- Handles open-ended queries well, ideal for summarization-style answers.

### 🔹 Extractive QA Model – `distilbert-base-cased-distilled-squad`

**Summary**:  
Extracts specific values (e.g., buyout sums, percentages, product counts) directly from summaries using span-based question answering.

**Justification**:  
- Lightweight and efficient with strong performance on SQuAD-like tasks.
- Ideal for pinpointing exact facts or numbers within text.
- Complements the generative model by offering high-precision extraction when exact answers are needed.


---

## 📁 Project Structure

```plaintext
├── data/
│   └── wildberries_mock_sales.csv          # Mock sales data
├── generate_text/
│   ├── text_generation.ipynb               # Notebook for generating text from sales data
│   └── wildberries.txt                     # Output: generated text summaries
├── README.md                               # This file
├── app.py                                  # Streamlit app for querying the assistant
├── rag_backend.py                          # Backend logic for retrieval-augmented generation
├── requirements.txt                        # Python dependencies
├── session_id.txt                          # Session tracking ID
├── test_queries.docx                       # Sample test queries
└── Credential/                             # 🔐 Contains authentication keys (NOT pushed to GitHub)
```

## ⚠️ **Important**:  
The folder named `Credential/` must be placed in the root directory. This folder includes sensitive credentials (e.g., API keys)

---

## 🚀 How to Run

1. **Set up virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate      # On Windows: venv\Scripts\activate
   ```
2. **Set up virtual environment**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Place the** ```Credential/``` **folder in the project root.**

4. **Run the Streamlit app:**:
   ```bash
   streamlit run app.py
   ```

---

## 🧠 Features

 - Converts structured sales metrics into fluent summaries
 - Includes retrieval-augmented generation (RAG) backend
 - Streamlit frontend for interactive user input
