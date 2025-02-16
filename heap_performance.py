import time
import random
import matplotlib.pyplot as plt
import numpy as np
from heap import RankingHeap
from typing import List, Tuple, Dict
from tqdm import tqdm

class HeapPerformanceAnalyzer:
    """Analyzes and visualizes RankingHeap performance"""
    
    def __init__(self):
        self.sizes = [100, 1000, 10000, 100000]
        self.results: Dict[str, Dict[int, float]] = {
            'insert': {},
            'peek': {},
            'get': {},
            'memory': {}
        }
        
    def generate_data(self, size: int) -> List[Tuple[float, str]]:
        """Generate test data of given size"""
        return [(random.random(), f"page_{i}") for i in range(size)]
        
    def measure_insert_performance(self, heap: RankingHeap, data: List[Tuple[float, str]]) -> float:
        """Measure insertion time"""
        start_time = time.time()
        for relevance, page in data:
            heap.add_page(relevance, page)
        return time.time() - start_time
        
    def measure_peek_performance(self, heap: RankingHeap, iterations: int = 100) -> float:
        """Measure peek operation time"""
        start_time = time.time()
        for _ in range(iterations):
            heap.peek_top_results(10)
        return (time.time() - start_time) / iterations
        
    def measure_get_performance(self, heap: RankingHeap, iterations: int = 100) -> float:
        """Measure get operation time"""
        start_time = time.time()
        for _ in range(iterations):
            heap.get_top_results(10)
        return (time.time() - start_time) / iterations
        
    def measure_memory(self, heap: RankingHeap) -> float:
        """Measure memory usage in MB"""
        import psutil
        import os
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
        
    def run_analysis(self):
        """Run performance analysis for different heap sizes"""
        print("Running performance analysis...")
        
        for size in tqdm(self.sizes, desc="Analyzing heap sizes"):
            # Generate test data
            data = self.generate_data(size)
            heap = RankingHeap(max_heap_size=size)
            
            # Measure insert performance
            self.results['insert'][size] = self.measure_insert_performance(heap, data)
            
            # Measure peek performance
            self.results['peek'][size] = self.measure_peek_performance(heap)
            
            # Measure get performance
            self.results['get'][size] = self.measure_get_performance(heap)
            
            # Measure memory usage
            self.results['memory'][size] = self.measure_memory(heap)
            
    def plot_results(self):
        """Generate performance visualization plots"""
        operations = ['insert', 'peek', 'get']
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('RankingHeap Performance Analysis', fontsize=16)
        
        # Plot time complexity
        for i, operation in enumerate(operations):
            ax = axes[i // 2, i % 2]
            times = [self.results[operation][size] for size in self.sizes]
            
            # Plot actual times
            ax.plot(self.sizes, times, 'b-', label='Actual')
            
            # Plot theoretical complexity
            if operation == 'insert':
                # O(log n)
                theoretical = [times[0] * np.log2(n) / np.log2(self.sizes[0]) for n in self.sizes]
            else:
                # O(k log n) for peek/get
                theoretical = [times[0] * np.log2(n) / np.log2(self.sizes[0]) for n in self.sizes]
                
            ax.plot(self.sizes, theoretical, 'r--', label='Theoretical')
            
            ax.set_xlabel('Heap Size')
            ax.set_ylabel('Time (seconds)')
            ax.set_title(f'{operation.capitalize()} Operation Performance')
            ax.legend()
            ax.grid(True)
            ax.set_xscale('log')
            ax.set_yscale('log')
            
        # Plot memory usage
        ax = axes[1, 1]
        memory = [self.results['memory'][size] for size in self.sizes]
        ax.plot(self.sizes, memory, 'g-', label='Actual')
        
        # Plot linear memory growth
        theoretical_memory = [memory[0] * n / self.sizes[0] for n in self.sizes]
        ax.plot(self.sizes, theoretical_memory, 'r--', label='Linear')
        
        ax.set_xlabel('Heap Size')
        ax.set_ylabel('Memory (MB)')
        ax.set_title('Memory Usage')
        ax.legend()
        ax.grid(True)
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig('heap_performance.png')
        plt.close()
        
    def print_summary(self):
        """Print performance summary"""
        print("\nPerformance Summary:")
        print("=" * 60)
        
        for size in self.sizes:
            print(f"\nHeap Size: {size}")
            print("-" * 40)
            print(f"Insert time: {self.results['insert'][size]:.6f} seconds")
            print(f"Peek time: {self.results['peek'][size]:.6f} seconds")
            print(f"Get time: {self.results['get'][size]:.6f} seconds")
            print(f"Memory usage: {self.results['memory'][size]:.2f} MB")
            
        print("\nComplexity Analysis:")
        print("-" * 40)
        print("Insert: O(log n)")
        print("Peek: O(k log n)")
        print("Get: O(k log n)")
        print("Memory: O(n)")
        
    def analyze(self):
        """Run complete analysis"""
        self.run_analysis()
        self.plot_results()
        self.print_summary()

def main():
    analyzer = HeapPerformanceAnalyzer()
    analyzer.analyze()

if __name__ == "__main__":
    main() 