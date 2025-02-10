import time
import memory_profiler
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple, Dict
import random
import string
from inverted_index import InvertedIndex

class PerformanceAnalyzer:
    def __init__(self):
        self.index = InvertedIndex(base_path="./perf_test_index", num_shards=4)
        
    def generate_test_documents(self, num_docs: int, words_per_doc: int) -> List[Tuple[int, str]]:
        """Generate test documents with controlled vocabulary"""
        vocabulary = [
            ''.join(random.choices(string.ascii_lowercase, k=random.randint(3, 10)))
            for _ in range(1000)  # Fixed vocabulary size
        ]
        
        documents = []
        for i in range(num_docs):
            words = random.choices(vocabulary, k=words_per_doc)
            documents.append((i, ' '.join(words)))
        return documents

    def measure_indexing_performance(self, doc_counts: List[int]) -> Dict[str, List[float]]:
        """Measure indexing time and memory usage for different document counts"""
        results = {
            'doc_counts': doc_counts,
            'index_times': [],
            'memory_usage': [],
            'throughput': []
        }
        
        for doc_count in doc_counts:
            # Generate test documents
            documents = self.generate_test_documents(doc_count, words_per_doc=100)
            
            # Measure indexing time
            start_time = time.time()
            self.index.add_documents(iter(documents))
            index_time = time.time() - start_time
            
            # Calculate throughput (docs/second)
            throughput = doc_count / index_time
            
            # Measure memory usage
            memory_usage = memory_profiler.memory_usage()[0]
            
            results['index_times'].append(index_time)
            results['memory_usage'].append(memory_usage)
            results['throughput'].append(throughput)
            
            # Clean up
            self.index = InvertedIndex(base_path="./perf_test_index", num_shards=4)
            
        return results

    def measure_search_performance(self, num_queries: int) -> Dict[str, List[float]]:
        """Measure search performance metrics"""
        # Generate and index test documents
        documents = self.generate_test_documents(10000, words_per_doc=100)
        self.index.add_documents(iter(documents))
        
        # Generate test queries
        queries = [
            ''.join(random.choices(string.ascii_lowercase, k=random.randint(3, 8)))
            for _ in range(num_queries)
        ]
        
        results = {
            'search_times': [],
            'result_counts': [],
            'latency_percentiles': []
        }
        
        # Measure search performance
        for query in queries:
            start_time = time.time()
            search_results = self.index.search(query)
            search_time = time.time() - start_time
            
            results['search_times'].append(search_time)
            results['result_counts'].append(len(search_results))
        
        # Calculate latency percentiles
        latencies = sorted(results['search_times'])
        results['latency_percentiles'] = {
            '50th': np.percentile(latencies, 50),
            '95th': np.percentile(latencies, 95),
            '99th': np.percentile(latencies, 99)
        }
        
        return results

    def plot_performance_metrics(self, index_results: Dict[str, List[float]], 
                               search_results: Dict[str, List[float]]):
        """Plot performance metrics"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Indexing time vs document count
        ax1.plot(index_results['doc_counts'], index_results['index_times'])
        ax1.set_xlabel('Number of Documents')
        ax1.set_ylabel('Indexing Time (s)')
        ax1.set_title('Indexing Performance')
        
        # Memory usage vs document count
        ax2.plot(index_results['doc_counts'], index_results['memory_usage'])
        ax2.set_xlabel('Number of Documents')
        ax2.set_ylabel('Memory Usage (MB)')
        ax2.set_title('Memory Scaling')
        
        # Search latency distribution
        ax3.hist(search_results['search_times'], bins=50)
        ax3.set_xlabel('Search Time (s)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Search Latency Distribution')
        
        # Throughput vs document count
        ax4.plot(index_results['doc_counts'], index_results['throughput'])
        ax4.set_xlabel('Number of Documents')
        ax4.set_ylabel('Documents/Second')
        ax4.set_title('Indexing Throughput')
        
        plt.tight_layout()
        plt.savefig('performance_analysis.png')
        plt.close()

def main():
    analyzer = PerformanceAnalyzer()
    
    # Test with increasing document counts
    doc_counts = [1000, 5000, 10000, 50000, 100000]
    index_results = analyzer.measure_indexing_performance(doc_counts)
    
    # Test search performance
    search_results = analyzer.measure_search_performance(1000)
    
    # Plot results
    analyzer.plot_performance_metrics(index_results, search_results)
    
    # Print performance summary
    print("\nPerformance Analysis Summary")
    print("=" * 50)
    
    print("\nIndexing Performance:")
    print(f"Max throughput: {max(index_results['throughput']):.2f} docs/second")
    print(f"Memory usage per 10k docs: {index_results['memory_usage'][2]/10:.2f} MB")
    
    print("\nSearch Performance:")
    print(f"Average latency: {np.mean(search_results['search_times'])*1000:.2f} ms")
    print(f"95th percentile latency: {search_results['latency_percentiles']['95th']*1000:.2f} ms")
    print(f"Average results per query: {np.mean(search_results['result_counts']):.2f}")

if __name__ == "__main__":
    main() 