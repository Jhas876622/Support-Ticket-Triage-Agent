import os
import json
import threading
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

class KnowledgeBaseRetriever:
    """
    Enterprise-Grade RAG Retriever:
    1. Thread-safe concurrency with re-entrant read/write locks (RLock).
    2. Category Metadata-Aware Re-Ranking & domain score boosting.
    3. MetricGuard confidence math: 0.75 * top_score + 0.25 * margin.
    4. Dual engine: lightweight scikit-learn TF-IDF or SentenceTransformers.
    5. Atomic file persistence to prevent file corruption.
    """
    def __init__(self, data_path: str = None, model_name: str = "all-MiniLM-L6-v2"):
        if data_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_path = os.path.join(base_dir, "data", "faq_kb.json")
        
        self.data_path = data_path
        self.model_name = model_name
        self.model = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.docs = []
        self.embeddings = None
        self._lock = threading.RLock()
        
        self.load_data()

    def _init_model(self):
        if self.model is None and HAS_SENTENCE_TRANSFORMERS:
            try:
                self.model = SentenceTransformer(self.model_name)
            except Exception as e:
                print(f"[INFO] SentenceTransformers unavailable ({e}), using scikit-learn TF-IDF engine.")
                self.model = None

    def load_data(self):
        with self._lock:
            if os.path.exists(self.data_path):
                with open(self.data_path, "r", encoding="utf-8") as f:
                    self.docs = json.load(f)
            else:
                self.docs = []
                
            self._build_index()

    def _build_index(self):
        with self._lock:
            if not self.docs:
                self.embeddings = None
                self.tfidf_matrix = None
                return
                
            texts = [
                f"{doc.get('title', '')} {doc.get('content', '')} {' '.join(doc.get('tags', []))}"
                for doc in self.docs
            ]

            self._init_model()
            if self.model:
                self.embeddings = self.model.encode(texts, normalize_embeddings=True)
            elif HAS_SKLEARN:
                self.tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
                self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            else:
                self.embeddings = None

    def add_doc(self, title: str, category: str, content: str, tags: List[str] = None) -> Dict[str, Any]:
        with self._lock:
            doc_id = f"faq_{len(self.docs) + 1:03d}"
            new_doc = {
                "id": doc_id,
                "title": title.strip(),
                "category": category.strip().lower(),
                "content": content.strip(),
                "tags": tags or []
            }
            self.docs.append(new_doc)
            
            # Atomic write to disk using temp file
            os.makedirs(os.path.dirname(self.data_path), exist_ok=True)
            temp_path = f"{self.data_path}.tmp"
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(self.docs, f, indent=2)
            os.replace(temp_path, self.data_path)
                
            self._build_index()
            return new_doc

    def search(self, query: str, top_k: int = 3, category: Optional[str] = None) -> Tuple[List[Dict[str, Any]], float]:
        """
        Thread-safe semantic search with optional Category Metadata Boosting.
        Returns: (ranked_docs_with_scores, confidence_score)
        """
        with self._lock:
            if not self.docs:
                return [], 0.0

            self._init_model()
            raw_scores = []

            # 1. Compute Base Similarity Scores
            if self.model and self.embeddings is not None:
                query_emb = self.model.encode([query], normalize_embeddings=True)[0]
                raw_scores = list(np.dot(self.embeddings, query_emb))
            elif HAS_SKLEARN and self.tfidf_matrix is not None and self.tfidf_vectorizer is not None:
                query_vec = self.tfidf_vectorizer.transform([query])
                raw_scores = list(cosine_similarity(query_vec, self.tfidf_matrix)[0])
            else:
                # Fallback keyword overlap
                query_words = set(query.lower().split())
                for doc in self.docs:
                    doc_text = f"{doc['title']} {doc['content']} {' '.join(doc.get('tags', []))}".lower()
                    doc_words = set(doc_text.split())
                    score = len(query_words.intersection(doc_words)) / (len(query_words) + 1e-5)
                    raw_scores.append(min(1.0, score * 1.5))

            # 2. Apply Category Metadata Boosting (+0.15 boost for in-category docs)
            scored_candidates = []
            for idx, doc in enumerate(self.docs):
                base_score = float(raw_scores[idx])
                boost = 0.0
                if category and doc.get("category", "").lower() == category.lower():
                    boost = 0.15  # Domain relevance boost
                final_score = min(1.0, round(base_score + boost, 4))
                doc_copy = dict(doc)
                doc_copy["base_similarity"] = round(base_score, 4)
                doc_copy["similarity_score"] = final_score
                doc_copy["category_boosted"] = boost > 0
                scored_candidates.append(doc_copy)

            # 3. Sort by final similarity score
            scored_candidates.sort(key=lambda x: x["similarity_score"], reverse=True)
            top_results = scored_candidates[:top_k]

            # 4. MetricGuard Confidence Math
            top_score = top_results[0]["similarity_score"] if top_results else 0.0
            second_score = top_results[1]["similarity_score"] if len(top_results) > 1 else 0.0
            margin = max(0.0, top_score - second_score)
            
            # Weighted formula: absolute similarity (75%) + disambiguation separation margin (25%)
            confidence = round(float(0.75 * top_score + 0.25 * margin), 4)
            return top_results, confidence
