import unittest
import time
import random
import string
import sys
import memory_profiler
from tire import Trie

class TestTrie(unittest.TestCase):
    def setUp(self):
        self.trie = Trie()
        
    def test_basic_operations(self):
        """Test basic trie operations"""
        # Test insertion and search
        test_words = ["hello", "help", "world", "word"]
        for word in test_words:
            self.trie.insert(word)
            
        for word in test_words:
            self.assertTrue(self.trie.search_prefix(word))
        
        # Test non-existent words
        self.assertFalse(self.trie.search_prefix("nothere"))
        self.assertFalse(self.trie.search_prefix("hel"))
        
    def test_case_sensitivity(self):
        """Test case-insensitive handling"""
        test_pairs = [
            ("Hello", "hello"),
            ("WORLD", "world"),
            ("MiXeD", "mixed")
        ]
        
        for original, lower in test_pairs:
            self.trie.insert(original)
            self.assertTrue(self.trie.search_prefix(lower))
            self.assertTrue(self.trie.search_prefix(original))
            
    def test_empty_and_edge_cases(self):
        """Test edge cases and boundary conditions"""
        # Empty string
        self.trie.insert("")
        self.assertFalse(self.trie.search_prefix(""))
        
        # Single character
        self.trie.insert("a")
        self.assertTrue(self.trie.search_prefix("a"))
        
        # Very long string
        long_string = "a" * 1000
        self.trie.insert(long_string)
        self.assertTrue(self.trie.search_prefix(long_string))
        
        # Special characters
        special = "!@#$%^&*()"
        self.trie.insert(special)
        self.assertTrue(self.trie.search_prefix(special))
        
    @memory_profiler.profile
    def test_memory_usage(self):
        """Test memory usage with large datasets"""
        # Generate random words
        words = set()
        for _ in range(100_000):
            length = random.randint(3, 10)
            word = ''.join(random.choices(string.ascii_lowercase, k=length))
            words.add(word)
            
        # Measure memory usage during insertion
        start_mem = memory_profiler.memory_usage()[0]
        for word in words:
            self.trie.insert(word)
        end_mem = memory_profiler.memory_usage()[0]
        
        # Print memory usage statistics
        print(f"Memory usage: {end_mem - start_mem:.2f} MiB")
        print(f"Average memory per word: {(end_mem - start_mem) / len(words):.6f} MiB")
        
    def test_performance_scaling(self):
        """Test performance scaling with dataset size"""
        sizes = [1000, 10000, 100000]
        times = {}
        
        for size in sizes:
            words = set()
            while len(words) < size:
                length = random.randint(3, 10)
                word = ''.join(random.choices(string.ascii_lowercase, k=length))
                words.add(word)
                
            # Test insertion time
            start_time = time.time()
            for word in words:
                self.trie.insert(word)
            insert_time = time.time() - start_time
            
            # Test search time
            start_time = time.time()
            for word in words:
                self.trie.search_prefix(word)
            search_time = time.time() - start_time
            
            times[size] = {
                'insert': insert_time,
                'search': search_time,
                'insert_per_word': insert_time / size,
                'search_per_word': search_time / size
            }
            
        # Print performance statistics
        for size, metrics in times.items():
            print(f"\nDataset size: {size}")
            print(f"Total insert time: {metrics['insert']:.4f}s")
            print(f"Total search time: {metrics['search']:.4f}s")
            print(f"Average insert time per word: {metrics['insert_per_word']*1000:.4f}ms")
            print(f"Average search time per word: {metrics['search_per_word']*1000:.4f}ms")
            
    def test_compression_effectiveness(self):
        """Test the effectiveness of path compression"""
        # Generate words with common prefixes
        prefixes = ["pre", "pro", "con", "com"]
        suffixes = ["ing", "ed", "s", "ly"]
        words = [p + s for p in prefixes for s in suffixes]
        
        # Insert words and check node count
        initial_nodes = self.trie._node_count
        for word in words:
            self.trie.insert(word)
            
        # Calculate compression ratio
        theoretical_nodes = sum(len(word) for word in words)
        actual_nodes = self.trie._node_count - initial_nodes
        compression_ratio = actual_nodes / theoretical_nodes
        
        print(f"\nCompression Statistics:")
        print(f"Theoretical nodes: {theoretical_nodes}")
        print(f"Actual nodes: {actual_nodes}")
        print(f"Compression ratio: {compression_ratio:.2%}")
        
    def test_autocomplete_stress(self):
        """Stress test autocomplete functionality"""
        # Generate a large dataset with varying frequencies
        words_with_freq = {}
        for _ in range(10000):
            length = random.randint(3, 10)
            word = ''.join(random.choices(string.ascii_lowercase, k=length))
            freq = random.randint(1, 100)
            words_with_freq[word] = freq
            
        # Insert words multiple times based on frequency
        for word, freq in words_with_freq.items():
            for _ in range(freq):
                self.trie.insert(word)
                
        # Test autocomplete with various prefixes
        prefixes = ["a", "th", "pre", "con"]
        for prefix in prefixes:
            start_time = time.time()
            results = self.trie.autocomplete(prefix, limit=10)
            end_time = time.time()
            
            print(f"\nAutocomplete for '{prefix}':")
            print(f"Time taken: {(end_time - start_time)*1000:.2f}ms")
            print(f"Results: {results}")
            
if __name__ == '__main__':
    unittest.main(verbosity=2) 