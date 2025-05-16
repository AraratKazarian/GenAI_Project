# rag_backend.py (fully updated with contextual memory, safe fallback, and cleaned logic)

import re
import os
import numpy as np
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from google.cloud import bigquery
import streamlit as st
from datetime import datetime, timedelta

# === Setup ===
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "Credentials/regal-autonomy-455818-u2-4a83b2648b3a.json"
bq_client = bigquery.Client()
BQ_TABLE = "regal-autonomy-455818-u2.GenAI.Sessions"

_MONTH_TO_MM = {
    "january": "01", "february": "02", "march": "03",
    "april": "04", "may": "05", "june": "06",
    "july": "07", "august": "08", "september": "09",
    "october": "10", "november": "11", "december": "12"
}

METRIC_COLUMNS = [
    'openCardCount', 'addToCartCount', 'ordersCount', 'ordersSumRub',
    'buyoutsCount', 'buyoutsSumRub', 'buyoutPercent',
    'addToCartConversion', 'cartToOrderConversion'
]

# === Session tracking ===
def get_session_id():
    if "session_id" not in st.session_state:
        if os.path.exists("session_id.txt"):
            with open("session_id.txt", "r") as f:
                sid = int(f.read().strip()) + 1
        else:
            sid = 1
        with open("session_id.txt", "w") as f:
            f.write(str(sid))
        st.session_state["session_id"] = sid
    return st.session_state["session_id"]

def log_to_bigquery(session_id: int, question: str, answer: str):
    row = [{"session_id": session_id, "user_prompt": question, "model_answer": answer}]
    errors = bq_client.insert_rows_json(BQ_TABLE, row)
    if not errors:
        st.session_state["last_question"] = question
        st.session_state["last_answer"] = answer
        print("✅ Logged to BigQuery")
    else:
        st.error(f"BigQuery insert failed: {errors}")

# === RAG System ===
class RAGSystem:
    def __init__(self, docs_path, embed_model="all-MiniLM-L6-v2", gen_model="google/flan-t5-base", device=0, df_path="fetch_data/wildberries_mock_sales.csv"):
        with open(docs_path, "r", encoding="utf-8") as f:
            self.docs = [line.strip() for line in f if line.strip()]
        self.embedder = SentenceTransformer(embed_model)
        embs = self.embedder.encode(self.docs, convert_to_numpy=True).astype(np.float32)
        self.index = faiss.IndexFlatL2(embs.shape[1])
        self.index.add(embs)
        self.df2 = pd.read_csv(df_path)
        self.gen_pipe = pipeline("text2text-generation", model=gen_model, tokenizer=gen_model, device=device)

    def _extract_date(self, ql):
        if any(k in ql for k in ["previous day", "the day before"]) and "last_date" in st.session_state:
            return (datetime.strptime(st.session_state["last_date"], "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
        elif "next day" in ql and "last_date" in st.session_state:
            return (datetime.strptime(st.session_state["last_date"], "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
        m_iso = re.search(r"(\d{4}-\d{2}-\d{2})", ql)
        if m_iso:
            return m_iso.group(1)
        m = re.search(r"(\w+)\s+(\d{1,2})", ql)
        if m:
            mm = _MONTH_TO_MM.get(m.group(1).lower())
            return f"2025-{mm}-{int(m.group(2)):02d}" if mm else None
        return None

    def _max_metric_for_product(self, product_id, metric):
        sub = self.df2[self.df2["productId"] == product_id]
        if sub.empty:
            return f"No data for product {product_id}."
        row = sub.loc[sub[metric].idxmax()]
        return f"Product {product_id} had the highest {metric} on {row['dt']}: {row[metric]}"

    def query(self, question, top_k=8):
        ql = question.lower()

        if re.search(r"\btell me about\s+(product\s*)?p\b(?!\d)", ql):
            return "Sorry, 'P' is not a valid product ID. Please specify a full product like 'P1', 'P2', etc."

        if "last_product" in st.session_state and "product" not in ql and not re.search(r"p\d+", ql):
            del st.session_state["last_product"]
        if "last_date" in st.session_state and not any(k in ql for k in ["day", r"\d{4}-\d{2}-\d{2}"]):
            del st.session_state["last_date"]

        date = self._extract_date(ql)
        metric_aliases = {
            "added to cart": "addToCartCount",
            "ordered": "ordersCount",
            "orders": "ordersCount",
            "views": "openCardCount",
            "buyouts": "buyoutsCount",
            "buyout rate": "buyoutPercent",
            "order value": "ordersSumRub",
        }

        product_match = re.search(r"(?:product\s*)?(p\d+)", ql)
        if not product_match and "last_product" in st.session_state:
            class MockMatch:  # fallback match class
                def group(self, _): return st.session_state["last_product"]
            product_match = MockMatch()

        if product_match:
            pid = product_match.group(1).upper()
            if not re.fullmatch(r"P\d+", pid):
                return f"'{pid}' is not a valid product ID. Use format like 'P1', 'P2'."
            st.session_state["last_product"] = pid
            if "tell me about" in ql:
                return self._summary_by_product(pid)
            for alias, col in metric_aliases.items():
                if "average" in ql and alias in ql:
                    return self._avg_metric_for_product(pid, col)
                if alias in ql:
                    if any(x in ql for x in ["when", "which day", "what day"]):
                        return self._product_with_max_metric(col)
                    return self._avg_metric_for_product(pid, col)
            for col in METRIC_COLUMNS:
                if col.lower() in ql or f"most {col.lower()}" in ql:
                    return self._max_metric_for_product(pid, col)

        for alias, col in metric_aliases.items():
            if alias in ql and any(x in ql for x in ["when", "which day", "what day"]):
                return self._product_with_max_metric(col)
            if re.search(fr"(most|highest).*{alias}|{alias}.*(most|highest)", ql):
                return self._product_with_max_metric(col, date)

        if date:
            return self._summary_by_date(date)

        qv = self.embedder.encode([question], convert_to_numpy=True).astype(np.float32)
        D, I = self.index.search(qv, top_k)
        if D[0][0] > 1.5:
            return "Sorry, I couldn't find relevant information to answer that question."

        top_docs = [self.docs[i] for i in I[0]]
        context = "\n".join(f"- {s}" for s in top_docs)
        prompt = f"Question: {question}\nContext:\n{context}\nAnswer concisely."
        tok = self.gen_pipe.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.gen_pipe.tokenizer.model_max_length).to(self.gen_pipe.device)
        out_ids = self.gen_pipe.model.generate(**tok, max_new_tokens=64, num_beams=4, early_stopping=True)
        return self.gen_pipe.tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()

    def _summary_by_date(self, date):
        st.session_state["last_date"] = date
        sub = self.df2[self.df2["dt"] == date]
        if sub.empty:
            return f"No data found for date {date}."
        lines = [f"Summary for all products on {date}:"]
        for _, row in sub.iterrows():
            parts = [f"Product {row['productId']}:"]
            for col in METRIC_COLUMNS:
                parts.append(f"  - {col}: {row[col]}")
            lines.append("\n".join(parts))
        return "\n\n".join(lines)

    def _summary_by_product(self, product_id):
        st.session_state["last_product"] = product_id
        sub = self.df2[self.df2["productId"] == product_id]
        if sub.empty:
            return f"No data found for product {product_id}."
        total = sub[METRIC_COLUMNS].sum(numeric_only=True)
        avg = sub[METRIC_COLUMNS].mean(numeric_only=True)
        lines = [f"Summary for product {product_id} across {len(sub)} days:"]
        for col in METRIC_COLUMNS:
            lines.append(f"- Total {col}: {total[col]:,.0f}, Average: {avg[col]:.2f}")
        return "\n".join(lines)

    def _avg_metric_for_product(self, product_id, metric):
        sub = self.df2[self.df2["productId"] == product_id]
        if sub.empty:
            return f"No data for product {product_id}."
        avg = sub[metric].mean()
        return f"Product {product_id} had an average {metric} of {avg:.2f}"

    def _product_with_max_metric(self, metric, date=None):
        sub = self.df2
        if date:
            sub = sub[sub["dt"] == date]
        if sub.empty:
            return "No data found."
        grouped = sub.groupby("dt")[metric].sum()
        max_date = grouped.idxmax()
        max_value = grouped[max_date]
        return f"{max_date} had the highest {metric}: {max_value}"

# === Interface ===
rag = RAGSystem(docs_path="generate_text/wildberries.txt")

def get_rag_answer(query: str) -> str:
    answer = rag.query(query)
    session_id = get_session_id()
    log_to_bigquery(session_id, query, answer)
    return answer

if __name__ == "__main__":
    while True:
        q = input("Ask a question (or 'exit'): ")
        if q.lower() == "exit":
            break
        print("Answer:", get_rag_answer(q))
