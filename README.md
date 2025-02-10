# Search Engine Components Implementation

This project implements core search engine components including a Trie data structure for autocomplete, an Inverted Index for full-text search, and a Ranking Heap for result scoring.

## Features

### Trie Implementation
- Path compression for memory efficiency
- LRU caching for frequent searches
- Case-insensitive search
- Autocomplete functionality
- Memory usage optimization

### Inverted Index
- Sharded storage for scalability
- Parallel processing with multi-threading
- Memory-efficient storage with disk offloading
- TF-IDF scoring for search relevance
- Phrase search capabilities
- Compressed storage using zlib

### Ranking Heap
- Efficient top-K results retrieval
- LRU result caching
- Memory-bounded operation
- Configurable size limits

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd search-engine-components
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running Tests
```bash


# Run specific test files
python -m unittest test_trie.py -v
python -m unittest test_inverted_index.py -v
python -m unittest test_heap.py -v
```

### Performance Analysis
```bash
# Run performance benchmarks
python benchmark_trie.py
python performance_analysis.py
```

### Example Usage

```python
# Using the Trie for autocomplete
from tire import Trie

trie = Trie()
trie.insert("hello")
trie.insert("help")
trie.insert("world")

results = trie.autocomplete("he", limit=5)
print(results)  # ['hello', 'help']

# Using the Inverted Index for search
from inverted_index import InvertedIndex

index = InvertedIndex()
documents = [
    (1, "the quick brown fox"),
    (2, "jumps over the lazy dog")
]
index.add_documents(iter(documents))

results = index.search("quick brown")
print(results)  # [(1, score)]

# Using the Ranking Heap
from heap import RankingHeap

heap = RankingHeap(cache_size=100)
heap.add_page(0.8, "page1")
heap.add_page(0.9, "page2")

top_results = heap.peek_top_results(2)
print(top_results)  # ['page2', 'page1']
```

## Performance Monitoring

The project includes comprehensive performance analysis tools:
- Memory usage tracking
- Execution time measurements
- Scalability testing
- Performance visualization

View generated performance graphs in:
- `performance_analysis.png`
- `insert_time_comparison.png`
- `search_time_comparison.png`
- `memory_comparison.png`

## Project Structure

```
.
├── tire.py                    # Trie implementation
├── inverted_index.py          # Inverted index implementation
├── heap.py                    # Ranking heap implementation
├── test_trie.py              # Trie tests
├── test_inverted_index.py    # Inverted index tests
├── test_heap.py              # Ranking heap tests
├── benchmark_trie.py         # Trie performance benchmarks
├── performance_analysis.py    # Overall performance analysis
└── requirements.txt          # Project dependencies
```

