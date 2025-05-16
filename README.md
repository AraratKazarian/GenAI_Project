# Smart Sales Assistant for Wildberries  
**Course**: DS 235 - Generative AI  
**Group Members**: Ararat Kazarian, Astghik Grigoryan  

---

## 📌 Project Overview

The **Smart Sales Assistant for Wildberries** is a generative AI tool designed to analyze mock sales data and generate human-readable summaries from key performance metrics. It enables quick insights for sales performance and decision-making in a scalable and interactive format.

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
