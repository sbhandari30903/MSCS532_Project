"""
Inverted Index Implementation with Sharding and Memory Management

This module implements a memory-efficient inverted index with the following features:
- Sharded storage for scalability
- Parallel processing for better performance
- Memory management with disk offloading
- TF-IDF scoring for search relevance
- Phrase search capabilities
- Compressed storage using zlib
"""

import re
from collections import defaultdict
from typing import List, Set, Dict, Tuple, Iterator, Optional
import math
import pickle
import zlib
import os
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
import threading
import logging

class IndexShard:
    """
    Represents a single shard of the inverted index.
    
    Each shard manages its own portion of the index and can independently:
    - Handle concurrent access with thread safety
    - Manage memory usage
    - Persist to disk when needed
    - Load data on demand
    """
    def __init__(self, shard_id: int, base_path: str):
        self.shard_id = shard_id
        self.base_path = base_path
        # Main index structure: word -> {doc_id -> [positions]}
        self.index: Dict[str, Dict[int, List[int]]] = defaultdict(dict)
        self.is_dirty = False  # Track if shard needs saving
        self._lock = threading.Lock()  # Thread safety
        
    def get_shard_path(self) -> str:
        """Generate filesystem path for this shard's data file"""
        return os.path.join(self.base_path, f"shard_{self.shard_id}.pkl.gz")
    
    def save(self) -> None:
        """
        Compress and save shard to disk.
        Uses zlib compression to reduce storage space.
        Only saves if data has been modified (is_dirty).
        """
        if not self.is_dirty:
            return
            
        with open(self.get_shard_path(), 'wb') as f:
            compressed = zlib.compress(pickle.dumps(self.index))
            f.write(compressed)
        self.is_dirty = False
            
    def load(self) -> None:
        """
        Load shard from disk if exists.
        Decompresses data using zlib.
        Creates empty index if no file exists.
        """
        path = self.get_shard_path()
        if os.path.exists(path):
            with open(path, 'rb') as f:
                compressed_data = f.read()
                self.index = pickle.loads(zlib.decompress(compressed_data))

class InvertedIndex:
    """
    Main inverted index implementation with sharding and parallel processing.
    
    Features:
    - Distributed storage across multiple shards
    - Parallel document processing
    - Memory-efficient storage
    - TF-IDF scoring
    - Phrase search support
    """
    def __init__(self, base_path: str = "./index_data", num_shards: int = 4):
        self.base_path = base_path
        self.num_shards = num_shards
        os.makedirs(base_path, exist_ok=True)
        
        # Metadata storage for document statistics
        self.doc_lengths: Dict[int, int] = {}  # Document length tracking
        self.total_docs = 0  # Total number of documents indexed
        self.batch_size = 1000  # Number of docs to process in parallel
        
        # Initialize shards for distributed storage
        self.shards: List[IndexShard] = [
            IndexShard(i, base_path) for i in range(num_shards)
        ]
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=num_shards)
        
        # Load existing index data if available
        self._load_metadata()
        
    def _get_shard_for_word(self, word: str) -> IndexShard:
        """
        Determine which shard should contain the word.
        Uses consistent hashing to distribute words across shards.
        """
        shard_id = hash(word) % self.num_shards
        return self.shards[shard_id]
        
    def preprocess_text(self, text: str) -> List[str]:
        """
        Clean and tokenize text for indexing.
        
        Performs:
        - Lowercase conversion
        - Special character removal
        - Tokenization
        - Empty token removal
        
        Args:
            text: Raw text to process
            
        Returns:
            List of cleaned tokens
        """
        if not text or not isinstance(text, str):
            return []
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-z\s]', '', text)
        
        # Split into words and remove empty strings
        words = [word.strip() for word in text.split() if word.strip()]
        
        return words

    def _process_document_batch(self, docs: List[Tuple[int, str]]) -> None:
        """
        Process a batch of documents in parallel.
        
        Implements two-phase processing:
        1. Collect words and positions for each shard
        2. Update shards in parallel
        
        Args:
            docs: List of (doc_id, content) pairs to process
        """
        word_to_shard_mapping: Dict[int, Dict[str, List[Tuple[int, int]]]] = defaultdict(lambda: defaultdict(list))
        
        # First pass: collect words and positions for each shard
        for doc_id, content in docs:
            words = self.preprocess_text(content)
            self.doc_lengths[doc_id] = len(words)
            
            for position, word in enumerate(words):
                shard_id = hash(word) % self.num_shards
                word_to_shard_mapping[shard_id][word].append((doc_id, position))
        
        # Second pass: update shards in parallel
        def update_shard(shard_id: int, word_data: Dict[str, List[Tuple[int, int]]]) -> None:
            shard = self.shards[shard_id]
            with shard._lock:
                for word, occurrences in word_data.items():
                    for doc_id, position in occurrences:
                        if doc_id not in shard.index[word]:
                            shard.index[word][doc_id] = []
                        shard.index[word][doc_id].append(position)
                shard.is_dirty = True
                
        # Execute shard updates in parallel
        futures = []
        for shard_id, word_data in word_to_shard_mapping.items():
            futures.append(self.executor.submit(update_shard, shard_id, word_data))
        
        # Wait for all updates to complete
        for future in futures:
            future.result()

    def add_documents(self, documents: Iterator[Tuple[int, str]]) -> None:
        """
        Add multiple documents to the index in batches.
        
        Features:
        - Batch processing for efficiency
        - Parallel processing of batches
        - Automatic metadata updates
        
        Args:
            documents: Iterator of (doc_id, content) pairs
        """
        batch = []
        
        for doc_id, content in documents:
            batch.append((doc_id, content))
            
            if len(batch) >= self.batch_size:
                self._process_document_batch(batch)
                self.total_docs += len(batch)
                batch = []
                
        if batch:
            self._process_document_batch(batch)
            self.total_docs += len(batch)
            
        self._save_metadata()

    def search(self, query: str, use_tfidf: bool = True, max_results: int = 100) -> List[Tuple[int, float]]:
        """
        Search with TF-IDF scoring and result limiting.
        
        Features:
        - TF-IDF scoring for relevance
        - Parallel term processing
        - Result limiting for efficiency
        
        Args:
            query: Search query string
            use_tfidf: Whether to use TF-IDF scoring
            max_results: Maximum number of results to return
            
        Returns:
            List of (doc_id, score) pairs, sorted by relevance
        """
        query_terms = self.preprocess_text(query)
        if not query_terms:
            return []

        scores: Dict[int, float] = defaultdict(float)
        
        # Process each term in parallel
        def process_term(term: str) -> Dict[int, float]:
            term_scores: Dict[int, float] = defaultdict(float)
            shard = self._get_shard_for_word(term)
            
            if term not in shard.index:
                return term_scores

            # Calculate IDF score
            idf = math.log(self.total_docs / len(shard.index[term]))
            
            for doc_id, positions in shard.index[term].items():
                if use_tfidf:
                    # Calculate TF-IDF score
                    tf = len(positions) / self.doc_lengths[doc_id]
                    term_scores[doc_id] = tf * idf
                else:
                    term_scores[doc_id] = 1
                    
            return term_scores

        # Execute term processing in parallel
        futures = [self.executor.submit(process_term, term) for term in query_terms]
        
        # Combine scores from all terms
        for future in futures:
            term_scores = future.result()
            for doc_id, score in term_scores.items():
                scores[doc_id] += score

        # Sort and limit results
        ranked_results = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return ranked_results[:max_results]

    def phrase_search(self, phrase: str) -> Set[int]:
        """
        Memory-efficient phrase search implementation.
        
        Features:
        - Position-based matching
        - Memory-efficient processing
        - Batch document processing
        
        Args:
            phrase: Exact phrase to search for
            
        Returns:
            Set of document IDs containing the phrase
        """
        words = self.preprocess_text(phrase)
        if not words:
            return set()

        # Get documents containing first word
        first_shard = self._get_shard_for_word(words[0])
        if words[0] not in first_shard.index:
            return set()
        
        candidate_docs = set(first_shard.index[words[0]].keys())

        # Filter documents containing all words
        for word in words[1:]:
            shard = self._get_shard_for_word(word)
            if word not in shard.index:
                return set()
            candidate_docs &= set(shard.index[word].keys())

        results = set()
        
        # Process candidates in batches to manage memory
        batch_size = 100
        for i in range(0, len(candidate_docs), batch_size):
            batch_docs = list(islice(candidate_docs, i, i + batch_size))
            
            for doc_id in batch_docs:
                positions = first_shard.index[words[0]][doc_id]
                
                for pos in positions:
                    is_phrase = True
                    for i, word in enumerate(words[1:], 1):
                        shard = self._get_shard_for_word(word)
                        if pos + i not in set(shard.index[word].get(doc_id, [])):
                            is_phrase = False
                            break
                    if is_phrase:
                        results.add(doc_id)
                        break

        return results

    def _save_metadata(self) -> None:
        """
        Save metadata and trigger shard saves.
        
        Saves:
        - Document lengths
        - Total document count
        - Individual shard data
        """
        metadata = {
            'doc_lengths': self.doc_lengths,
            'total_docs': self.total_docs
        }
        
        metadata_path = os.path.join(self.base_path, 'metadata.pkl.gz')
        with open(metadata_path, 'wb') as f:
            compressed = zlib.compress(pickle.dumps(metadata))
            f.write(compressed)
            
        # Save all shards
        for shard in self.shards:
            shard.save()

    def _load_metadata(self) -> None:
        """
        Load metadata and shards from disk.
        
        Loads:
        - Document statistics
        - Shard data
        Creates empty structures if no data exists.
        """
        metadata_path = os.path.join(self.base_path, 'metadata.pkl.gz')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'rb') as f:
                compressed_data = f.read()
                metadata = pickle.loads(zlib.decompress(compressed_data))
                self.doc_lengths = metadata['doc_lengths']
                self.total_docs = metadata['total_docs']
                
        # Load all shards
        for shard in self.shards:
            shard.load()

    def cleanup(self) -> None:
        """
        Clean up resources before shutdown.
        
        Performs:
        - Metadata saving
        - Thread pool shutdown
        """
        self._save_metadata()
        self.executor.shutdown()
    
