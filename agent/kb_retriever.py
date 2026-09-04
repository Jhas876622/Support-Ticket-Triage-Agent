import os
import json
import numpy as np
from typing import List, Dict, Any, Tuple

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

class KnowledgeBaseRetriever:
    """
    RAG retriever using Sentence Transformers embeddings (MetricGuard pattern)
    with cosine similarity search and retrieval confidence scoring.
    """
    def __init__(self, data_path: str = None, model_name: str = "all-MiniLM-L6-v2"):
        if data_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_path = os.path.join(base_dir, "data", "faq_kb.json")
        
        self.data_path = data_path
        self.model_name = model_name
        self.model = None
        self.docs = []
        self.embeddings = None
        
        self.load_data()

    def _init_model(self):
        if self.model is None and HAS_SENTENCE_TRANSFORMERS:
            try:
                # Load lightweight embeddings model
                self.model = SentenceTransformer(self.model_name)
            except Exception as e:
                print(f"[WARN] Failed to load SentenceTransformer model ({e}), falling back to keyword/TF-IDF similarity.")
                self.model = None

    def load_data(self):
        if os.path.exists(self.data_path):
            with open(self.data_path, "r", encoding="utf-8") as f:
                self.docs = json.load(f)
        else:
            self.docs = []
            
        self._build_index()

    def _build_index(self):
        if not self.docs:
            self.embeddings = None
            return
            
        self._init_model()
        if self.model:
            texts = [
                f"{doc.get('title', '')} - {doc.get('content', '')} Tags: {' '.join(doc.get('tags', []))}"
                for doc in self.docs
            ]
            self.embeddings = self.model.encode(texts, normalize_embeddings=True)
        else:
            self.embeddings = None

    def add_doc(self, title: str, category: str, content: str, tags: List[str] = None) -> Dict[str, Any]:
        doc_id = f"faq_{len(self.docs) + 1:03d}"
        new_doc = {
            "id": doc_id,
            "title": title,
            "category": category,
            "content": content,
            "tags": tags or []
        }
        self.docs.append(new_doc)
        
        # Save to disk
        os.makedirs(os.path.dirname(self.data_path), exist_ok=True)
        with open(self.data_path, "w", encoding="utf-8") as f:
            json.dump(self.docs, f, indent=2)
            
        self._build_index()
        return new_doc

    def search(self, query: str, top_k: int = 3) -> Tuple[List[Dict[str, Any]], float]:
        """
        Searches knowledge base for top matching docs.
        Returns: (matched_docs_with_scores, confidence_score)
        """
        if not self.docs:
            return [], 0.0

        self._init_model()

        if self.model and self.embeddings is not None:
            query_emb = self.model.encode([query], normalize_embeddings=True)[0]
            # Cosine similarity since vectors are normalized
            similarities = np.dot(self.embeddings, query_emb)
            
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                score = float(similarities[idx])
                doc = dict(self.docs[idx])
                doc["similarity_score"] = round(score, 4)
                results.append(doc)
                
            top_score = results[0]["similarity_score"] if results else 0.0
            second_score = results[1]["similarity_score"] if len(results) > 1 else 0.0
            
            # Confidence metric based on top similarity and margin over second best match
            margin = max(0.0, top_score - second_score)
            confidence = round(float(0.75 * top_score + 0.25 * margin), 4)
            return results, confidence

        else:
            # Fallback keyword/tag intersection similarity calculation
            query_words = set(query.lower().split())
            scored_docs = []
            for doc in self.docs:
                doc_text = f"{doc['title']} {doc['content']} {' '.join(doc.get('tags', []))}".lower()
                doc_words = set(doc_text.split())
                intersection = query_words.intersection(doc_words)
                score = len(intersection) / (len(query_words) + 1e-5)
                # Map score to [0, 1] range
                score = min(1.0, round(score * 1.5, 4))
                doc_copy = dict(doc)
                doc_copy["similarity_score"] = score
                scored_docs.append(doc_copy)
                
            scored_docs.sort(key=lambda x: x["similarity_score"], reverse=True)
            top_results = scored_docs[:top_k]
            top_score = top_results[0]["similarity_score"] if top_results else 0.0
            return top_results, top_score
