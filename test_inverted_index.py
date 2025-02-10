import unittest
import tempfile
import shutil
import os
import time
import random
import string
import psutil
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple
from inverted_index import InvertedIndex

def get_process_memory() -> float:
    """Get current process memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

class TestInvertedIndex(unittest.TestCase):
    def setUp(self):
        """Create a temporary directory for index storage"""
        self.test_dir = tempfile.mkdtemp()
        self.index = InvertedIndex(base_path=self.test_dir, num_shards=4)

    def tearDown(self):
        """Clean up temporary files"""
        self.index.cleanup()
        shutil.rmtree(self.test_dir)

    def test_basic_functionality(self):
        """Test basic index operations"""
        documents = [
            (1, "the quick brown fox"),
            (2, "jumps over the lazy dog"),
            (3, "the quick brown dog sleeps")
        ]
        
        # Test document addition
        self.index.add_documents(iter(documents))
        
        # Test simple search
        results = self.index.search("quick", use_tfidf=False)
        self.assertEqual(len(results), 2)
        self.assertIn(1, [doc_id for doc_id, _ in results])
        self.assertIn(3, [doc_id for doc_id, _ in results])
        
        # Test phrase search
        phrase_results = self.index.phrase_search("quick brown")
        self.assertEqual(len(phrase_results), 2)
        self.assertIn(1, phrase_results)
        self.assertIn(3, phrase_results)

    def test_tfidf_scoring(self):
        """Test TF-IDF scoring correctness"""
        documents = [
            (1, "test word test word test"),  # Higher TF for 'test'
            (2, "test word"),                 # Lower TF for 'test'
            (3, "word word word")             # No 'test'
        ]
        
        self.index.add_documents(iter(documents))
        results = self.index.search("test", use_tfidf=True)
        
        # Document 1 should rank higher than Document 2
        self.assertTrue(len(results) >= 2)
        doc1_score = next(score for doc_id, score in results if doc_id == 1)
        doc2_score = next(score for doc_id, score in results if doc_id == 2)
        self.assertGreater(doc1_score, doc2_score)

    def test_concurrent_operations(self):
        """Test concurrent document additions and searches"""
        def generate_document(doc_id: int) -> Tuple[int, str]:
            words = ['test', 'concurrent', 'operation', 'document']
            content = ' '.join(random.choices(words, k=10))
            return (doc_id, content)

        # Add documents concurrently
        with ThreadPoolExecutor(max_workers=4) as executor:
            documents = [generate_document(i) for i in range(100)]
            self.index.add_documents(iter(documents))

            # Perform concurrent searches
            search_futures = [
                executor.submit(self.index.search, "test")
                for _ in range(10)
            ]
            
            # Verify all searches complete without errors
            for future in search_futures:
                results = future.result()
                self.assertIsNotNone(results)

    def test_memory_efficiency(self):
        """Test memory usage during large document processing"""
        def generate_large_document(doc_id: int) -> Tuple[int, str]:
            # Generate ~1KB of text
            words = ''.join(random.choices(string.ascii_lowercase + ' ', k=1000))
            return (doc_id, words)

        # Measure initial memory
        initial_memory = get_process_memory()
        
        # Process documents in batches to test memory growth
        batch_sizes = [100, 500, 1000]
        memory_measurements = []
        
        for batch_size in batch_sizes:
            documents = [generate_large_document(i) for i in range(batch_size)]
            self.index.add_documents(iter(documents))
            
            # Measure memory after batch processing
            current_memory = get_process_memory()
            memory_measurements.append((batch_size, current_memory))
            
            # Perform some searches to test memory stability
            for _ in range(50):
                query = ''.join(random.choices(string.ascii_lowercase, k=3))
                self.index.search(query)
            
            # Measure memory after searches
            post_search_memory = get_process_memory()
            
            # Verify memory growth is reasonable (less than 2x per 10x docs)
            memory_growth = post_search_memory - initial_memory
            docs_ratio = batch_size / 100  # ratio compared to first batch
            self.assertLess(memory_growth / docs_ratio, initial_memory * 2,
                          f"Memory growth too high for batch size {batch_size}")
            
        # Print memory usage statistics
        print("\nMemory Usage Statistics:")
        print("Batch Size | Memory Usage (MB) | Growth Factor")
        print("-" * 50)
        for i, (batch_size, memory) in enumerate(memory_measurements):
            growth = memory - initial_memory
            print(f"{batch_size:10d} | {memory:15.2f} | {growth:13.2f}")

    def test_persistence(self):
        """Test index persistence and recovery"""
        documents = [
            (1, "test persistence"),
            (2, "recovery test")
        ]
        
        # Add documents and force save
        self.index.add_documents(iter(documents))
        self.index.cleanup()
        
        # Create new index instance and verify data
        new_index = InvertedIndex(base_path=self.test_dir, num_shards=4)
        results = new_index.search("persistence")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], 1)

    def test_stress_large_documents(self):
        """Stress test with large documents"""
        def generate_stress_document(doc_id: int, size_kb: int) -> Tuple[int, str]:
            words = []
            while len(' '.join(words)) < size_kb * 1024:
                words.append(''.join(random.choices(string.ascii_lowercase, k=random.randint(3, 10))))
            return (doc_id, ' '.join(words))

        start_time = time.time()
        
        # Generate and index 100 documents of 100KB each
        documents = [generate_stress_document(i, 100) for i in range(100)]
        self.index.add_documents(iter(documents))
        
        indexing_time = time.time() - start_time
        print(f"\nIndexing time for large documents: {indexing_time:.2f} seconds")
        
        # Test search performance
        search_times = []
        for _ in range(100):
            query = ''.join(random.choices(string.ascii_lowercase, k=3))
            start_time = time.time()
            results = self.index.search(query)
            search_times.append(time.time() - start_time)
        
        avg_search_time = sum(search_times) / len(search_times)
        print(f"Average search time: {avg_search_time:.4f} seconds")
        
        self.assertLess(avg_search_time, 0.1, "Search performance below threshold")

    def test_edge_cases(self):
        """Test edge cases and error handling"""
        # Empty document
        self.index.add_documents(iter([(1, "")]))
        results = self.index.search("")
        self.assertEqual(len(results), 0, "Empty search query should return no results")
        
        # Empty word search in non-empty document
        self.index.add_documents(iter([(2, "test document")]))
        results = self.index.search("")
        self.assertEqual(len(results), 0, "Empty search query should return no results")
        
        # Very long word
        long_word = "a" * 1000
        self.index.add_documents(iter([(3, long_word)]))
        results = self.index.search(long_word)
        self.assertEqual(len(results), 1, "Should find document with very long word")
        
        # Special characters
        special_chars = "!@#$%^&*()_+-=[]{}|;:'\",.<>?/\\"
        self.index.add_documents(iter([(4, special_chars)]))
        results = self.index.search(special_chars)
        self.assertEqual(len(results), 0, "Special characters should be removed in preprocessing")
        
        # Unicode characters
        unicode_text = "Hello 世界"
        self.index.add_documents(iter([(5, unicode_text)]))
        results = self.index.search("世界")
        self.assertEqual(len(results), 0, "Non-ASCII characters should be handled gracefully")
        
        # Numbers
        numeric_text = "123 456 789"
        self.index.add_documents(iter([(6, numeric_text)]))
        results = self.index.search("123")
        self.assertEqual(len(results), 0, "Numbers should be removed in preprocessing")
        
        # Case sensitivity tests
        self.index.add_documents(iter([
            (7, "UPPERCASE WORDS"),
            (8, "lowercase words"),
            (9, "MiXeD cAsE wOrDs")
        ]))
        
        # Test each case variation
        uppercase_results = self.index.search("UPPERCASE")
        lowercase_results = self.index.search("uppercase")
        mixedcase_results = self.index.search("UpPerCase")
        
        # All searches should return the same document
        self.assertEqual(len(uppercase_results), 1, "Case insensitive search should work")
        self.assertEqual(len(lowercase_results), 1, "Case insensitive search should work")
        self.assertEqual(len(mixedcase_results), 1, "Case insensitive search should work")
        
        # Verify it's the correct document
        self.assertEqual(uppercase_results[0][0], 7, "Should find the correct document")
        self.assertEqual(lowercase_results[0][0], 7, "Should find the correct document")
        self.assertEqual(mixedcase_results[0][0], 7, "Should find the correct document")

    def test_scalability(self):
        """Test scalability with increasing dataset sizes"""
        sizes = [100, 1000, 10000]
        times = []
        
        for size in sizes:
            documents = [
                (i, f"document number {i} with some random words " + 
                    ' '.join(random.choices(string.ascii_lowercase, k=10)))
                for i in range(size)
            ]
            
            start_time = time.time()
            self.index.add_documents(iter(documents))
            index_time = time.time() - start_time
            
            # Measure search time
            start_time = time.time()
            for _ in range(100):
                self.index.search("random")
            search_time = (time.time() - start_time) / 100
            
            times.append((size, index_time, search_time))
            
            # Clean up for next iteration
            shutil.rmtree(self.test_dir)
            os.makedirs(self.test_dir)
            self.index = InvertedIndex(base_path=self.test_dir, num_shards=4)
        
        # Print scalability results
        print("\nScalability Test Results:")
        print("Dataset Size | Index Time | Avg Search Time")
        print("-" * 45)
        for size, index_time, search_time in times:
            print(f"{size:11d} | {index_time:10.2f}s | {search_time:13.4f}s")
        
        # Verify sub-linear search time growth
        if len(times) >= 2:
            ratio = times[-1][2] / times[0][2]  # Compare largest to smallest search time
            size_ratio = sizes[-1] / sizes[0]
            self.assertLess(ratio, size_ratio, 
                "Search time should scale sub-linearly with dataset size")

if __name__ == '__main__':
    unittest.main(verbosity=2) 