import time
import random
import string
import memory_profiler
import matplotlib.pyplot as plt
from tire import Trie as OptimizedTrie

class BasicTrie:
    """Basic Trie implementation for comparison"""
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False

    def insert(self, word):
        node = self
        for char in word.lower():
            if char not in node.children:
                node.children[char] = BasicTrie()
            node = node.children[char]
        node.is_end_of_word = True

    def search(self, word):
        node = self
        for char in word.lower():
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end_of_word

def generate_dataset(size, min_len=3, max_len=10):
    """Generate random words for testing"""
    return [''.join(random.choices(string.ascii_lowercase, 
            k=random.randint(min_len, max_len))) for _ in range(size)]

class TrieBenchmark:
    def __init__(self):
        self.datasets = {
            'small': generate_dataset(1000),
            'medium': generate_dataset(10000),
            'large': generate_dataset(100000)
        }
        self.results = {}

    @memory_profiler.profile
    def measure_memory(self, trie_class, dataset):
        """Measure memory usage for insertion"""
        trie = trie_class()
        mem_before = memory_profiler.memory_usage()[0]
        for word in dataset:
            trie.insert(word)
        mem_after = memory_profiler.memory_usage()[0]
        return mem_after - mem_before

    def measure_performance(self, trie_class, dataset):
        """Measure insertion and search times"""
        trie = trie_class()
        
        # Measure insertion time
        start_time = time.time()
        for word in dataset:
            trie.insert(word)
        insert_time = time.time() - start_time
        
        # Measure search time
        start_time = time.time()
        for word in dataset:
            trie.search(word) if isinstance(trie, BasicTrie) else trie.search_prefix(word)
        search_time = time.time() - start_time
        
        return {
            'insert_time': insert_time,
            'search_time': search_time,
            'insert_per_word': insert_time / len(dataset),
            'search_per_word': search_time / len(dataset)
        }

    def run_benchmarks(self):
        implementations = {
            'Basic': BasicTrie,
            'Optimized': OptimizedTrie
        }
        
        for impl_name, impl_class in implementations.items():
            self.results[impl_name] = {}
            for size_name, dataset in self.datasets.items():
                print(f"\nBenchmarking {impl_name} Trie with {size_name} dataset...")
                
                # Measure performance
                perf_metrics = self.measure_performance(impl_class, dataset)
                
                # Measure memory
                mem_usage = self.measure_memory(impl_class, dataset)
                
                self.results[impl_name][size_name] = {
                    **perf_metrics,
                    'memory': mem_usage,
                    'memory_per_word': mem_usage / len(dataset)
                }

    def plot_results(self):
        """Generate performance comparison plots"""
        metrics = ['insert_time', 'search_time', 'memory']
        dataset_sizes = [len(ds) for ds in self.datasets.values()]
        
        for metric in metrics:
            plt.figure(figsize=(10, 6))
            for impl in self.results:
                values = [self.results[impl][size][metric] 
                         for size in self.datasets.keys()]
                plt.plot(dataset_sizes, values, marker='o', label=impl)
                
            plt.xlabel('Dataset Size')
            plt.ylabel(metric.replace('_', ' ').title())
            plt.title(f'{metric.replace("_", " ").title()} Comparison')
            plt.legend()
            plt.grid(True)
            plt.savefig(f'{metric}_comparison.png')
            plt.close()

    def print_summary(self):
        """Print detailed performance summary"""
        print("\nPerformance Summary:")
        print("=" * 80)
        
        for impl in self.results:
            print(f"\n{impl} Trie Implementation:")
            print("-" * 40)
            
            for size in self.datasets:
                metrics = self.results[impl][size]
                print(f"\nDataset Size: {len(self.datasets[size])}")
                print(f"Insert time: {metrics['insert_time']:.4f}s")
                print(f"Search time: {metrics['search_time']:.4f}s")
                print(f"Memory usage: {metrics['memory']:.2f} MiB")
                print(f"Average insert time per word: {metrics['insert_per_word']*1000:.4f}ms")
                print(f"Average search time per word: {metrics['search_per_word']*1000:.4f}ms")
                print(f"Average memory per word: {metrics['memory_per_word']:.6f} MiB")

if __name__ == '__main__':
    benchmark = TrieBenchmark()
    benchmark.run_benchmarks()
    benchmark.plot_results()
    benchmark.print_summary() 