# Search Engine Components Implementation

This project implements a full-featured search engine with Wikipedia article search capabilities, combining Trie data structure for autocomplete, Inverted Index for full-text search, and a Ranking Heap for result scoring.

## Features

### Search Engine (main.py)
- Wikipedia article search and indexing
- Full-text search with TF-IDF scoring
- Phrase search capabilities
- Query autocomplete suggestions
- Interactive command-line interface
- Progress tracking for article loading

### Core Components
#### Trie Implementation
- Path compression for memory efficiency
- LRU caching for frequent searches
- Case-insensitive search
- Autocomplete functionality

#### Inverted Index
- Sharded storage for scalability
- Parallel processing with multi-threading
- Memory-efficient storage with disk offloading
- TF-IDF scoring for search relevance

#### Ranking Heap
- Efficient top-K results retrieval
- LRU result caching
- Memory-bounded operation
- Configurable size limits

## Installation

1. Clone the repository:

bash
git clone https://github.com/yourusername/search-engine.git
cd search-engine

2. Create and activate virtual environment:

bash
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate

3. Install dependencies:

bash
pip install -r requirements.txt

## Usage

### Running the Search Engine

bash
python main.py

This will:
1. Load 100 random Wikipedia articles
2. Start an interactive search interface
3. Allow you to:
   - Perform full-text searches
   - Search for exact phrases
   - Get autocomplete suggestions

### Search Interface Options
1. **Search**: Full-text search with ranked results
2. **Phrase Search**: Find exact matches of phrases
3. **Autocomplete**: Get search suggestions as you type
4. **Exit**: Close the application

### Example Usage
```
Search Engine Interface
1. Search
2. Phrase Search
3. Autocomplete
4. Exit

Enter your choice (1-4): 1
Enter search query: python programming
Found 3 results:
1. Introduction to Python
   Python is a high-level programming language...

2. Programming Paradigms
   Python supports multiple programming paradigms...
```

## Performance Monitoring

The project includes comprehensive performance analysis tools:
- Memory usage tracking
- Execution time measurements
- Scalability testing
- Performance visualization


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
python performance_analysis_inverted_index.py
python heap_performance.py
```

View generated performance graphs in:
- `performance_analysis.png`
- `insert_time_comparison.png`
- `search_time_comparison.png`
- `memory_comparison.png`
- `heap_performance.png`

## Project Structure
```
.
├── main.py                    # Search engine implementation with Wikipedia integration
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

## Dependencies
- matplotlib: Visualization
- numpy: Numerical operations
- psutil: System monitoring
- memory_profiler: Memory analysis
- requests: Wikipedia API access
- tqdm: Progress bars
- pytest: Testing


