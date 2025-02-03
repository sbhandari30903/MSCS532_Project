import heapq

class RankingHeap:
    def __init__(self):
        # Initialize an empty list to represent the heap
        self.heap = []

    def add_page(self, relevance, page):
        # Push a tuple of negative relevance (for max-heap) and page onto the heap
        heapq.heappush(self.heap, (-relevance, page))

    def get_top_results(self, k):
        # Retrieve the top k pages by popping from the heap
        return [heapq.heappop(self.heap)[1] for _ in range(min(k, len(self.heap)))]