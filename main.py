from heap import RankingHeap
from inverted_index import InvertedIndex
from tire import Trie
from typing import List, Dict, Set, Tuple
import re
import requests
from tqdm import tqdm

class SearchEngine:
    """
    Unified search engine combining Inverted Index, Trie, and Ranking Heap
    for full-text search with autocomplete and ranked results.
    """
    def __init__(self, max_results: int = 10):
        self.index = InvertedIndex()
        self.trie = Trie()
        self.ranking_heap = RankingHeap(max_heap_size=max_results)
        self.doc_store: Dict[int, str] = {}  # Store document content
        self.doc_titles: Dict[int, str] = {}  # Store document titles
        self.current_doc_id = 0
        
    def add_document(self, title: str, content: str) -> int:
        """
        Add a document to the search engine.
        
        Args:
            title: Document title
            content: Document content
            
        Returns:
            doc_id: Unique document identifier
        """
        self.current_doc_id += 1
        doc_id = self.current_doc_id
        
        # Store document
        self.doc_store[doc_id] = content
        self.doc_titles[doc_id] = title
        
        # Index document content
        self.index.add_documents([(doc_id, content)])
        
        # Add title words to trie for autocomplete
        words = set(re.findall(r'\w+', title.lower()))
        for word in words:
            self.trie.insert(word)
            
        return doc_id
        
    def search(self, query: str, max_results: int = 10) -> List[Dict]:
        """
        Perform full-text search with ranked results.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            
        Returns:
            List of dictionaries containing document info and relevance scores
        """
        # Get matching documents with TF-IDF scores
        matches = self.index.search(query, use_tfidf=True)
        
        # Clear previous rankings
        self.ranking_heap = RankingHeap(max_heap_size=max_results)
        
        # Add results to ranking heap
        for doc_id, score in matches:
            self.ranking_heap.add_page(score, doc_id)
            
        # Get top ranked results
        top_doc_ids = self.ranking_heap.get_top_results(max_results)
        
        # Format results
        results = []
        for doc_id in top_doc_ids:
            # Get document snippet (first 2000 characters)
            content = self.doc_store[doc_id]
            snippet = content[:1000] + "..." if len(content) > 1000 else content
            
            results.append({
                'doc_id': doc_id,
                'title': self.doc_titles[doc_id],
                'snippet': snippet
            })
            
        return results
        
    def autocomplete(self, prefix: str, limit: int = 5) -> List[str]:
        """
        Get autocomplete suggestions for search queries.
        
        Args:
            prefix: Query prefix to autocomplete
            limit: Maximum number of suggestions
            
        Returns:
            List of autocomplete suggestions
        """
        return self.trie.autocomplete(prefix, limit)
        
    def phrase_search(self, phrase: str, max_results: int = 10) -> List[Dict]:
        """
        Search for exact phrases.
        
        Args:
            phrase: Exact phrase to search for
            max_results: Maximum number of results
            
        Returns:
            List of matching documents
        """
        doc_ids = self.index.phrase_search(phrase)
        results = []
        
        for doc_id in list(doc_ids)[:max_results]:
            content = self.doc_store[doc_id]
            snippet = content[:1000] + "..." if len(content) > 1000 else content
            
            results.append({
                'doc_id': doc_id,
                'title': self.doc_titles[doc_id],
                'snippet': snippet
            })
            
        return results

class SearchInterface:
    """Command-line interface for the search engine"""
    def __init__(self):
        self.search_engine = SearchEngine()
        
    def load_wikipedia_articles(self, limit: int = 100):
        """Load Wikipedia articles using the Wikipedia API."""
        print(f"\nLoading {limit} Wikipedia articles...")
        
        # Wikipedia API endpoint
        api_url = "https://en.wikipedia.org/w/api.php"
        
        params = {
            "action": "query",
            "format": "json",
            "list": "random",
            "rnlimit": min(500, limit),  # API limit is 500
            "rnnamespace": 0  # Main articles only
        }
        
        articles_loaded = 0
        with tqdm(total=limit, desc="Loading articles") as pbar:
            while articles_loaded < limit:
                response = requests.get(api_url, params=params)
                data = response.json()
                
                for article in data["query"]["random"]:
                    # Get full article content
                    content_params = {
                        "action": "query",
                        "format": "json",
                        "prop": "extracts",
                        "exintro": 1,
                        "explaintext": 1,
                        "pageids": article["id"]
                    }
                    
                    content_response = requests.get(api_url, params=content_params)
                    content_data = content_response.json()
                    
                    page_id = str(article["id"])
                    if "query" in content_data and "pages" in content_data["query"]:
                        content = content_data["query"]["pages"][page_id]["extract"]
                        self.search_engine.add_document(article["title"], content)
                        articles_loaded += 1
                        pbar.update(1)
                        
                    if articles_loaded >= limit:
                        break
                        
        print(f"\nLoaded {articles_loaded} Wikipedia articles.")
            
    def run(self):
        """Run the interactive search interface"""
        # Load Wikipedia articles
        self.load_wikipedia_articles(limit=1000)
        
        while True:
            print("\nSearch Engine Interface")
            print("1. Search")
            print("2. Phrase Search")
            print("3. Autocomplete")
            print("4. Exit")
            
            choice = input("Enter your choice (1-4): ")
            
            if choice == '1':
                query = input("Enter search query: ")
                results = self.search_engine.search(query)
                self._display_results(results)
                
            elif choice == '2':
                phrase = input("Enter exact phrase to search: ")
                results = self.search_engine.phrase_search(phrase)
                self._display_results(results)
                
            elif choice == '3':
                prefix = input("Enter prefix for autocomplete: ")
                suggestions = self.search_engine.autocomplete(prefix, 10)
                print("\nSuggestions:")
                for suggestion in suggestions:
                    print(f"- {suggestion}")
                    
            elif choice == '4':
                print("Goodbye!")
                break
                
            else:
                print("Invalid choice. Please try again.")
                
    def _display_results(self, results: List[Dict]):
        """Helper method to display search results"""
        if not results:
            print("\nNo results found.")
            return
            
        print(f"\nFound {len(results)} results:")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result['title']}")
            print(f"   {result['snippet']}")

if __name__ == "__main__":
    # Run the search interface
    interface = SearchInterface()
    interface.run()