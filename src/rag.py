"""Minimal, from-scratch RAG pipeline: PDF -> chunks -> BGE embeddings -> FAISS + BM25 hybrid
retrieval (reciprocal rank fusion) -> LLM answer with page-level citations."""
import os, re, requests
import numpy as np

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
NOT_FOUND = "I couldn't find this in the uploaded documents."

def split_text(text, size=800, overlap=150):
    """Sentence-aware chunking with character overlap."""
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks, cur = [], ""
    for s in sentences:
        if cur and len(cur) + len(s) > size:
            chunks.append(cur)
            cur = cur[-overlap:] + " " + s
        else:
            cur = (cur + " " + s).strip()
    if cur:
        chunks.append(cur)
    return chunks

def rrf(rank_lists, k=60):
    """Reciprocal Rank Fusion over several ranked lists of chunk ids."""
    scores = {}
    for ranks in rank_lists:
        for pos, cid in enumerate(ranks):
            scores[cid] = scores.get(cid, 0) + 1 / (k + pos + 1)
    return sorted(scores, key=scores.get, reverse=True)

class RAGStore:
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        import faiss
        self.embedder = SentenceTransformer(EMBED_MODEL)
        self.index = faiss.IndexFlatIP(self.embedder.get_sentence_embedding_dimension())
        self.chunks, self.bm25, self.docs = [], None, set()

    def add_pdf(self, name, data: bytes):
        import fitz
        from rank_bm25 import BM25Okapi
        if name in self.docs:
            return 0
        new = []
        for i, page in enumerate(fitz.open(stream=data, filetype="pdf")):
            for c in split_text(page.get_text()):
                if len(c) > 40:
                    new.append({"doc": name, "page": i + 1, "text": c})
        if not new:
            return 0
        emb = self.embedder.encode([c["text"] for c in new], normalize_embeddings=True,
                                   batch_size=32).astype("float32")
        self.index.add(emb)
        self.chunks += new
        self.docs.add(name)
        self.bm25 = BM25Okapi([c["text"].lower().split() for c in self.chunks])
        return len(new)

    def search(self, query, k=4, pool=20):
        if not self.chunks:
            return [], 0.0
        q = self.embedder.encode([QUERY_PREFIX + query], normalize_embeddings=True).astype("float32")
        scores, ids = self.index.search(q, min(pool, len(self.chunks)))
        dense = [int(i) for i in ids[0] if i >= 0]
        bm = self.bm25.get_scores(query.lower().split())
        sparse = [int(i) for i in np.argsort(bm)[::-1][:pool]]
        top = rrf([dense, sparse])[:k]
        return [self.chunks[i] for i in top], float(scores[0][0])   # best cosine sim = relevance gate

def build_prompt(question, hits, history):
    ctx = "\n\n".join(f"[{h['doc']} | p.{h['page']}]\n{h['text']}" for h in hits)
    hist = "\n".join(f"{r.upper()}: {t}" for r, t in history[-6:])
    return (f"You are a document assistant. Answer ONLY using the context below. "
            f"If the answer is not in the context, reply exactly: \"{NOT_FOUND}\" "
            f"Cite sources inline as [document, p.N].\n\n"
            f"CONTEXT:\n{ctx}\n\nCONVERSATION SO FAR:\n{hist}\n\nQUESTION: {question}\nANSWER:")

class LLMError(Exception):
    pass

def call_llm(prompt, provider="gemini", api_key=None):
    import time
    key = (api_key or os.getenv("GEMINI_API_KEY", "")).strip()
    if not key:
        raise LLMError("No Gemini API key configured.")
    models = [os.getenv("GEMINI_MODEL", "gemini-2.5-flash"), "gemini-3.5-flash", "gemini-2.5-flash-lite"]
    last = None
    for model in dict.fromkeys(models):
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        for attempt in range(2):
            r = requests.post(url, headers={"x-goog-api-key": key},
                              json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=60)
            if r.status_code in (429, 500, 503):      # busy / rate-limited -> retry, then next model
                last = r
                time.sleep(2 * (attempt + 1))
                continue
            if r.status_code in (400, 401, 403):      # key problem: no point retrying
                try:
                    msg = r.json()["error"]["message"]
                except Exception:
                    msg = r.text[:200]
                raise LLMError(f"Gemini rejected the API key ({r.status_code}): {msg}")
            if r.status_code == 404:                  # model name retired -> try next model
                last = r
                break
            r.raise_for_status()
            return r.json()["candidates"][0]["content"]["parts"][0]["text"]
    raise LLMError(f"Gemini is busy or unavailable right now (last status {getattr(last, 'status_code', '?')}). Try again in a minute.")

def answer(store, question, history, provider, api_key, min_score=0.40):
    hits, best = store.search(question)
    if not hits or best < min_score:          # out-of-scope guard -> no hallucination
        return NOT_FOUND, []
    return call_llm(build_prompt(question, hits, history), provider, api_key), hits
