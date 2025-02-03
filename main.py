from heap import RankingHeap
from inverted_index import InvertedIndex
from tire import Trie

if __name__ == "__main__":
    # Demonstration of Inverted Index
    doc1 = """The skills and knowledge gained from this 
    assignment are directly applicable to a range of modern 
    software technologies. Whether you're optimizing algorithms
    for large-scale data processing frameworks like Apache 
    Hadoop or Spark, enhancing the performance of search algorithms
    in web services, or developing resource-efficient applications
    for mobile and embedded systems, the principles of Quicksort
    and its analysis will be invaluable."""

    doc2 = """In this project, you will apply your knowledge 
    of data structures and algorithm analysis to design, 
    implement, and optimize data structures for a specific 
    real-world application using Python. Your project will 
    include coding the data structures, running performance 
    tests, analyzing results, and optimizing the structures for 
    efficiency. The final submission will demonstrate your 
    ability to translate theoretical concepts into practical, 
    high-performance code and will include both a detailed 
    report and a presentation."""

    index = InvertedIndex()
    index.add_document(1, doc1)
    index.add_document(2, doc2)
    print(index.search("Optimizing"))  # Output: {1, 2}

    # Demonstration of Trie
    trie = Trie()
    trie.insert("structures")
    trie.insert("performance")
    trie.insert("petros")
    print(trie.search_prefix("st"))  # Output: True
    print(trie.search_prefix("en"))  # Output: False
    print(trie.autocomplete("pe"))  # Returns performance and petros

    # Demonstration of Ranking Heap 
    heap = RankingHeap()
    heap.add_page(10, "Page A")
    heap.add_page(12, "Page B")
    heap.add_page(7, "Page C")
    print(heap.get_top_results(1))  # Output: ['Page B']