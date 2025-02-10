import heapq
from typing import List, Tuple, Any, Optional
from collections import deque

class RankingHeap:
    def __init__(self, cache_size: int = 100, max_heap_size: Optional[int] = None):
        # Initialize an empty list to represent the heap
        self.heap: List[Tuple[float, Any]] = []
        self.cache_size = cache_size
        self.cached_results: dict = {}
        self.max_heap_size = max_heap_size
        # LRU cache for recently accessed k values
        self.cache_queue = deque(maxlen=cache_size)

    def add_page(self, relevance: float, page: Any) -> Optional[Tuple[float, Any]]:
        """
        Add a page with O(log n) time complexity.
        Returns the removed item if heap exceeds max_size, None otherwise.
        """
        if self.max_heap_size and len(self.heap) >= self.max_heap_size:
            # Find the item with lowest relevance (least negative)
            lowest_idx = 0
            for i in range(len(self.heap)):
                if self.heap[i][0] > self.heap[lowest_idx][0]:
                    lowest_idx = i
                    
            # If new relevance is higher than the lowest
            if -relevance < self.heap[lowest_idx][0]:
                # Get the lowest relevance item
                lowest = self.heap[lowest_idx]
                # Replace it with the new item
                self.heap[lowest_idx] = (-relevance, page)
                # Restore heap property
                heapq.heapify(self.heap)
                self.cached_results.clear()
                # Return the removed item with positive relevance
                return (-lowest[0], lowest[1])
            return (relevance, page)  # Don't add if new item has lower relevance
        
        # If we haven't reached max size, just add the item
        heapq.heappush(self.heap, (-relevance, page))
        self.cached_results.clear()
        return None

    def peek_top_results(self, k: int) -> List[Any]:
        """Get top k results without modifying heap - O(n log k) first time, O(1) if cached."""
        if k in self.cached_results:
            # Update LRU cache order
            self.cache_queue.remove(k)
            self.cache_queue.append(k)
            return self.cached_results[k][:k]
            
        heap_copy = self.heap.copy()
        results = [heapq.heappop(heap_copy)[1] for _ in range(min(k, len(heap_copy)))]
        
        # Manage cache size using LRU policy
        if k <= self.cache_size:
            if len(self.cached_results) >= self.cache_size:
                # Remove least recently used cache entry
                oldest_k = self.cache_queue[0]
                del self.cached_results[oldest_k]
                self.cache_queue.remove(oldest_k)
            
            self.cached_results[k] = results
            self.cache_queue.append(k)
            
        return results
    
    def get_top_results(self, k: int) -> List[Any]:
        """Remove and return top k results - O(k log n)."""
        self.cached_results.clear()  # Clear cache since heap is modified
        return [heapq.heappop(self.heap)[1] for _ in range(min(k, len(self.heap)))]
    
    def size(self) -> int:
        """Return current heap size - O(1)."""
        return len(self.heap)
    
    def clear_cache(self) -> None:
        """Clear the cache to free memory."""
        self.cached_results.clear()
        self.cache_queue.clear()