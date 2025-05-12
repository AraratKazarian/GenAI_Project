import re
import numpy as np
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Hardcoded month-to-MM mapping
_MONTH_TO_MM = {
    "january": "01", "february": "02", "march": "03",
    "april":   "04", "may":      "05", "june":  "06",
    "july":    "07", "august":   "08", "september": "09",
    "october": "10", "november":"11", "december":  "12"
}

# Load your CSV for special handlers
df2 = pd.read_csv("fetch_data/wildberries_data.csv")

class RAGSystem:
    def __init__(
        self,
        docs_path: str,
        embed_model: str = "all-MiniLM-L6-v2",
        gen_model:   str = "google/flan-t5-base",
        device:      int = 0   # GPU=0, CPU=-1
    ):
        # Load one-line summaries
        with open(docs_path, "r", encoding="utf-8") as f:
            self.docs = [line.strip() for line in f if line.strip()]

        # Build embeddings + FAISS index
        self.embedder = SentenceTransformer(embed_model)
        embs = self.embedder.encode(self.docs, convert_to_numpy=True).astype(np.float32)
        self.index = faiss.IndexFlatL2(embs.shape[1])
        self.index.add(embs)

        # Local pipelines
        self.gen_pipe = pipeline(
            "text2text-generation",
            model=gen_model,
            tokenizer=gen_model,
            device=device
        )
        self.ext_pipe = pipeline(
            "question-answering",
            model="distilbert-base-cased-distilled-squad",
            tokenizer="distilbert-base-cased-distilled-squad",
            device=device
        )

    def retrieve(self, question: str, top_k: int = 8):
        """Return the top_k most similar summaries."""
        qv = self.embedder.encode([question], convert_to_numpy=True).astype(np.float32)
        _, I = self.index.search(qv, top_k)
        return [self.docs[i] for i in I[0]]

    def _handle_minmax_orders(self, question: str, summaries):
        """
        Handle "smallest/largest number of orders" by
        deferring to df2 directly via idxmin/idxmax.
        """
        ql = question.lower()
        if "smallest" in ql or "lowest" in ql:
            return df2.loc[df2["ordersCount"].idxmin(), "dt"]
        else:
            return df2.loc[df2["ordersCount"].idxmax(), "dt"]

    def _handle_maxmin_conversion(self, question: str, summaries):
        """Handle highest/lowest conversion-rate questions."""
        ql = question.lower()
        if "cart-to-order conversion" in ql:
            pat = r"On (\d{4}-\d{2}-\d{2}),.*?cart-to-order conversion rate was (\d+)%"
        else:
            pat = r"On (\d{4}-\d{2}-\d{2}),.*?add-to-cart conversion rate was (\d+)%"
        pairs = []
        for s in summaries:
            m = re.search(pat, s)
            if m:
                pairs.append((m.group(1), int(m.group(2))))
        if not pairs:
            return None
        return (
            min(pairs, key=lambda x: x[1])[0]
            if "lowest" in ql else
            max(pairs, key=lambda x: x[1])[0]
        )

    def _handle_numeric(self, question: str):
        """Handle date-specific numeric lookups."""
        ql = question.lower()
        # ISO date?
        m_date = re.search(r"(\d{4}-\d{2}-\d{2})", ql)
        if m_date:
            date = m_date.group(1)
        else:
            # "Tell me about May 5" style
            m_md = re.search(r"tell me about (\w+)\s*(\d{1,2})", ql)
            if not m_md:
                return None
            mon, day = m_md.group(1).lower(), int(m_md.group(2))
            mm = _MONTH_TO_MM.get(mon)
            if not mm:
                return None
            date = f"2025-{mm}-{day:02d}"

        # Find the matching summary
        summary = next((s for s in self.docs if date in s), "")
        if not summary:
            return None

        # Specific extractions
        if "added to the cart" in ql:
            m = re.search(r"(\d+) items were added to the cart", summary)
            return m.group(1) if m else None

        if "buyouts sum" in ql:
            m = re.search(r"buyouts occurred, worth (\d+) rubles", summary)
            return m.group(1) if m else None

        if "buyout percentage" in ql:
            m = re.search(r"buyout percentage of (\d+)%", summary)
            return f"{m.group(1)}%" if m else None

        if "orders sum" in ql or "orders worth" in ql:
            m = re.search(r"orders worth (\d+) rubles", summary)
            return m.group(1) if m else None

        # Fallback extractive QA
        out = self.ext_pipe(question=question, context=summary)
        return out.get("answer", "").strip()

    def _handle_summary(self, question: str):
        """Return the one-line summary for a given date or 'Tell me about...'."""
        ql = question.lower()
        m_iso = re.search(r"(\d{4}-\d{2}-\d{2})", ql)
        if m_iso:
            date = m_iso.group(1)
        else:
            m = re.search(r"tell me about (\w+)\s*(\d{1,2})", ql)
            if not m:
                return None
            mon, dd = m.group(1).lower(), int(m.group(2))
            mm = _MONTH_TO_MM.get(mon)
            if not mm:
                return None
            date = f"2025-{mm}-{dd:02d}"
        return next((s for s in self.docs if date in s), None)

    def query(self, question: str, top_k: int = 8):
        """
        Dispatches to:
          1) _handle_minmax_orders
          2) _handle_maxmin_conversion
          3) _handle_numeric
          4) _handle_summary
          5) fallback generative RAG
        """
        summaries = self.retrieve(question, top_k)
        ql = question.lower()

        # 1) Number of orders min/max
        if "smallest" in ql or "largest" in ql:
            ans = self._handle_minmax_orders(question, summaries)
            if ans:
                return ans

        # 2) Conversion-rate min/max
        if "highest" in ql or "lowest" in ql:
            ans = self._handle_maxmin_conversion(question, summaries)
            if ans:
                return ans

        # 3) Numeric lookups
        if any(k in ql for k in ("added to the cart", "buyouts sum", "buyout percentage", "orders sum", "orders worth")):
            ans = self._handle_numeric(question)
            if ans:
                return ans

        # 4) Summary / Tell me about
        if ql.startswith("summarize") or ql.startswith("tell me about"):
            ans = self._handle_summary(question)
            if ans:
                return ans

        # 5) Fallback generative RAG
        context = "\n".join(f"- {s}" for s in summaries)
        prompt  = f"Question: {question}\nSummaries:\n{context}\nAnswer concisely."
        tok     = self.gen_pipe.tokenizer(
            prompt, return_tensors="pt", truncation=True,
            max_length=self.gen_pipe.tokenizer.model_max_length
        ).to(self.gen_pipe.device)
        out_ids = self.gen_pipe.model.generate(
            **tok, max_new_tokens=32, num_beams=4, early_stopping=True
        )
        return self.gen_pipe.tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()

# instantiate once
rag = RAGSystem(docs_path="generate_text/wildberries.txt")

def get_rag_answer(query: str) -> str:
    """Single-call function for your UI."""
    return rag.query(query)

if __name__ == "__main__":
    while True:
        q = input("Ask a question (or 'exit'): ")
        if q.lower() == "exit":
            break
        print("Answer:", get_rag_answer(q))
