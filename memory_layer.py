from typing import Dict, List

import os
import pickle
import torch

import numpy as np
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def simple_tokenize(text):
    return word_tokenize(text)


class HybridRetriever:
    """Hybrid retrieval system combining BM25 and semantic search."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', alpha: float = 0.5):
        """Initialize the hybrid retriever.
        
        Args:
            model_name: Name of the SentenceTransformer model to use
            alpha: Weight for combining BM25 and semantic scores (0 = only BM25, 1 = only semantic)
        """
        self.model = SentenceTransformer(model_name)
        self.alpha = alpha
        self.bm25 = None
        self.corpus = []
        self.embeddings = None
        self.document_ids = {}  # Map document content to its index
        
    
    def save(self, retriever_cache_file: str, retriever_cache_embeddings_file: str):
        """Save retriever state to disk"""
        
        # Save embeddings using numpy
        if self.embeddings is not None:
            np.save(retriever_cache_embeddings_file, self.embeddings)
            
        # Save everything else using pickle
        state = {
            'alpha': self.alpha,
            'bm25': self.bm25,
            'corpus': self.corpus,
            'document_ids': self.document_ids,
            'model_name': 'all-MiniLM-L6-v2'  # Default value for model name
        }
        
        # Try to get the actual model name if possible
        try:
            model_name = getattr(self, "model_name", None)

            if model_name is None:
                raise RuntimeError(
                    "Cannot determine embedding model name when saving retriever cache. "
                    "Please set self.model_name in SimpleEmbeddingRetriever.__init__."
                )

            state["model_name"] = model_name
        except (AttributeError, KeyError):
            pass
            
        with open(retriever_cache_file, 'wb') as f:
            pickle.dump(state, f)
            
    @classmethod
    def load(cls, retriever_cache_file: str, retriever_cache_embeddings_file: str):
        """Load retriever state from disk"""
        # Load the pickled state
        with open(retriever_cache_file, 'rb') as f:
            state = pickle.load(f)
            
        # Create new instance
        retriever = cls(model_name=state['model_name'], alpha=state['alpha'])
        retriever.bm25 = state['bm25']
        retriever.corpus = state['corpus']
        retriever.document_ids = state.get('document_ids', {})
        
        # Load embeddings from numpy file if it exists
        if retriever_cache_embeddings_file.exists():
            retriever.embeddings = np.load(retriever_cache_embeddings_file)
            
        return retriever
    
    @classmethod
    def load_from_local_memory(cls, memories: Dict, model_name: str, alpha: float) -> bool:
        """Load retriever state from memory"""
        all_docs = [", ".join(m.keywords) for m in memories.values()] #[m.content for m in memories.values()]
        retriever = cls(model_name, alpha)
        retriever.add_documents(all_docs)
        return retriever
    
    def add_documents(self, documents: List[str]) -> bool:
        """One-time Add documents to both BM25 and semantic index"""
        if not documents:
            return
            
        # Tokenize for BM25
        tokenized_docs = [doc.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs)
        
        # Create embeddings
        self.embeddings = self.model.encode(documents)
        self.corpus = documents
        doc_idx = 0
        for document in documents:
            self.document_ids[document] = doc_idx
            doc_idx += 1

        return True

    def add_document(self, document: str) -> bool:
        """Add a single document to the retriever.
        
        Args:
            document: Text content to add
            
        Returns:
            bool: True if document was added, False if it was already present
        """
        # Check if document already exists
        if document in self.document_ids:
            return False
            
        # Add to corpus and get index
        doc_idx = len(self.corpus)
        self.corpus.append(document)
        self.document_ids[document] = doc_idx
        
        # Update BM25
        if self.bm25 is None:
            # First document, initialize BM25
            tokenized_corpus = [simple_tokenize(document)]
            self.bm25 = BM25Okapi(tokenized_corpus)
        else:
            # Add to existing BM25
            tokenized_doc = simple_tokenize(document)
            self.bm25.add_document(tokenized_doc)
        
        # Update embeddings
        doc_embedding = self.model.encode([document], convert_to_tensor=True)
        if self.embeddings is None:
            self.embeddings = doc_embedding
        else:
            self.embeddings = torch.cat([self.embeddings, doc_embedding])
            
        return True
        
    def retrieve(self, query: str, k: int = 5) -> List[int]:
        """Retrieve documents using hybrid scoring"""
        if not self.corpus:
            return []
            
        # Get BM25 scores
        tokenized_query = query.lower().split()
        bm25_scores = np.array(self.bm25.get_scores(tokenized_query))
        
        # Normalize BM25 scores if they exist
        if len(bm25_scores) > 0:
            bm25_scores = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-6)
        
        # Get semantic scores
        query_embedding = self.model.encode([query])[0]
        semantic_scores = cosine_similarity([query_embedding], self.embeddings)[0]
        
        # Combine scores
        hybrid_scores = self.alpha * bm25_scores + (1 - self.alpha) * semantic_scores
        
        # Get top k indices
        k = min(k, len(self.corpus))
        top_k_indices = np.argsort(hybrid_scores)[-k:][::-1]
        return top_k_indices.tolist()

class SimpleEmbeddingRetriever:
    """Simple retrieval system using only text embeddings."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize the simple embedding retriever.
        
        Args:
            model_name: Name of the SentenceTransformer model to use
        """
        self.model = SentenceTransformer(model_name)
        self.corpus = []
        self.embeddings = None
        self.document_ids = {}  # Map document content to its index
        # Show the effective SentenceTransformer/device selected by the local-model patch.
        print(
            "[DEBUG_QWEN_COMPARE] RETRIEVER_EMBED_INIT "
            f"requested_model={model_name} "
            f"effective_model={getattr(self.model, 'model_name_or_path', None) or model_name} "
            f"device={getattr(self.model, 'device', 'unknown')} "
            f"embedding_dim={self.model.get_sentence_embedding_dimension()}"
        )
        
    def add_documents(self, documents: List[str]):
        """Add documents to the retriever."""
        char_lengths = [len(str(document)) for document in documents]
        token_estimates = [len(str(document).split()) for document in documents]
        # A documents_count of one during consolidation confirms batch-size=1 encoding.
        print(
            "[DEBUG_QWEN_COMPARE] RETRIEVER_EMBED_CALL "
            f"operation=add_documents device={getattr(self.model, 'device', 'unknown')} "
            f"documents_count={len(documents)} corpus_before={len(self.corpus)} "
            f"single_document_encode={len(documents) == 1} "
            f"chars_min={min(char_lengths) if char_lengths else 0} "
            f"chars_avg={(sum(char_lengths) / len(char_lengths)) if char_lengths else 0:.2f} "
            f"chars_max={max(char_lengths) if char_lengths else 0} "
            f"tokens_est_min={min(token_estimates) if token_estimates else 0} "
            f"tokens_est_avg={(sum(token_estimates) / len(token_estimates)) if token_estimates else 0:.2f} "
            f"tokens_est_max={max(token_estimates) if token_estimates else 0} cache_hit=False"
        )
        # Reset if no existing documents
        if not self.corpus:
            self.corpus = documents
            # print("documents", documents, len(documents))
            self.embeddings = self.model.encode(documents)
            self.document_ids = {doc: idx for idx, doc in enumerate(documents)}
        else:
            # Append new documents
            start_idx = len(self.corpus)
            self.corpus.extend(documents)
            new_embeddings = self.model.encode(documents)
            if self.embeddings is None:
                self.embeddings = new_embeddings
            else:
                self.embeddings = np.vstack([self.embeddings, new_embeddings])
            for idx, doc in enumerate(documents):
                self.document_ids[doc] = start_idx + idx
        print(
            "[DEBUG_QWEN_COMPARE] RETRIEVER_EMBED_DONE "
            f"operation=add_documents documents_count={len(documents)} corpus_after={len(self.corpus)} "
            f"embedding_rows={0 if self.embeddings is None else len(self.embeddings)}"
        )
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, float]]:
        """Search for similar documents using cosine similarity.
        
        Args:
            query: Query text
            k: Number of results to return
            
        Returns:
            List of dicts with document text and score
        """
        if not self.corpus:
            return []
        print(
            "[DEBUG_QWEN_COMPARE] RETRIEVER_SEARCH "
            f"query_chars={len(str(query))} query_tokens_est={len(str(query).split())} "
            f"requested_top_k={k} candidate_documents={len(self.corpus)} device={getattr(self.model, 'device', 'unknown')}"
        )
        # print("corpus", len(self.corpus), self.corpus)
        # Encode query
        query_embedding = self.model.encode([query])[0]
        
        # Calculate cosine similarities
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        # Get top k results
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        
            
        return top_k_indices
        
    def save(self, retriever_cache_file: str, retriever_cache_embeddings_file: str):
        """Save retriever state to disk"""
        print(
            "[DEBUG_QWEN_COMPARE] RETRIEVER_CACHE_SAVE "
            f"cache_hit=False corpus_count={len(self.corpus)} "
            f"embedding_rows={0 if self.embeddings is None else len(self.embeddings)} "
            f"state_file={retriever_cache_file} embeddings_file={retriever_cache_embeddings_file}"
        )
        # Save embeddings using numpy
        if self.embeddings is not None:
            np.save(retriever_cache_embeddings_file, self.embeddings)
        
        # Save other attributes
        state = {
            'corpus': self.corpus,
            'document_ids': self.document_ids
        }
        with open(retriever_cache_file, 'wb') as f:
            pickle.dump(state, f)
    
    def load(self, retriever_cache_file: str, retriever_cache_embeddings_file: str):
        """Load retriever state from disk"""
        print(
            "[DEBUG_QWEN_COMPARE] RETRIEVER_CACHE_LOAD "
            f"state_exists={os.path.exists(retriever_cache_file)} "
            f"embeddings_exists={os.path.exists(retriever_cache_embeddings_file)} "
            f"state_file={retriever_cache_file} embeddings_file={retriever_cache_embeddings_file}"
        )
        print(f"Loading retriever from {retriever_cache_file} and {retriever_cache_embeddings_file}")
        
        # Load embeddings
        if os.path.exists(retriever_cache_embeddings_file):
            print(f"Loading embeddings from {retriever_cache_embeddings_file}")
            self.embeddings = np.load(retriever_cache_embeddings_file)
            print(f"Embeddings shape: {self.embeddings.shape}")
        else:
            print(f"Embeddings file not found: {retriever_cache_embeddings_file}")
        
        # Load other attributes
        if os.path.exists(retriever_cache_file):
            print(f"Loading corpus from {retriever_cache_file}")
            with open(retriever_cache_file, 'rb') as f:
                state = pickle.load(f)
                self.corpus = state['corpus']
                self.document_ids = state['document_ids']
                print(f"Loaded corpus with {len(self.corpus)} documents")
        else:
            print(f"Corpus file not found: {retriever_cache_file}")
            
        print(
            "[DEBUG_QWEN_COMPARE] RETRIEVER_CACHE_LOAD_DONE "
            f"cache_hit={bool(self.corpus) and self.embeddings is not None} corpus_count={len(self.corpus)} "
            f"embedding_rows={0 if self.embeddings is None else len(self.embeddings)}"
        )
        return self
