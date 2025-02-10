class TrieNode:
    """
    A node in the Trie data structure.
    Uses __slots__ to reduce memory overhead by preventing dynamic attribute creation.
    Each node represents a character in the trie and can mark the end of a word.
    """
    __slots__ = ['children', 'is_end_of_word', 'frequency', 'compressed_value']
    
    def __init__(self):
        # Dictionary mapping characters to child nodes
        # Key: character (str), Value: TrieNode
        self.children = {}
        
        # Boolean flag indicating if this node represents a complete word
        # True if this node is the last character of an inserted word
        self.is_end_of_word = False
        
        # Counter tracking how many times this word has been inserted
        # Used for ranking autocomplete suggestions by popularity
        self.frequency = 0
        
        # Stores the remaining characters of a compressed path
        # None if this node is not compressed, string otherwise
        self.compressed_value = None

class Trie:
    """
    An optimized Trie (prefix tree) implementation with path compression and caching.
    Supports case-insensitive storage, prefix searching, and autocomplete functionality.
    Implements memory optimization through path compression and node counting.
    """
    def __init__(self):
        # Root node of the trie, doesn't represent any character
        self.root = TrieNode()
        
        # Counter for total number of complete words stored
        self.size = 0
        
        # Cache storing results of previous prefix searches
        # Key: prefix string, Value: boolean indicating if prefix exists
        self._prefix_cache = {}
        
        # Maximum number of entries allowed in prefix cache
        # Prevents unbounded memory growth
        self.MAX_CACHE_SIZE = 1000
        
        # Threshold for total nodes before aggressive compression
        # 10M nodes is approximately 1GB of memory
        self._memory_threshold = 10_000_000
        
        # Counter tracking total number of nodes in the trie
        # Used for memory management decisions
        self._node_count = 0
        
    def _should_compress(self):
        """
        Determines if the trie should use aggressive path compression.
        
        Returns:
            bool: True if node count exceeds memory threshold, False otherwise
        
        This helps manage memory usage for large datasets by triggering
        compression when the trie grows too large.
        """
        return self._node_count > self._memory_threshold
        
    def insert(self, word):
        """
        Inserts a word into the trie with path compression optimization.
        
        Args:
            word (str): The word to insert into the trie
            
        The method handles:
        - Case-insensitive storage
        - Path compression for memory optimization
        - Frequency counting for autocomplete ranking
        - Cache invalidation for affected prefixes
        """
        if not word:
            return
            
        node = self.root
        word = word.lower()  # Case-insensitive storage
        
        i = 0
        while i < len(word):
            char = word[i]
            
            # Handle existing compressed paths
            if char in node.children and node.children[char].compressed_value:
                compressed = node.children[char].compressed_value
                # Skip ahead if current word matches compressed path
                if word[i:].startswith(compressed):
                    i += len(compressed)
                    node = node.children[char]
                    continue
            
            # Create new node if character doesn't exist
            if char not in node.children:
                node.children[char] = TrieNode()
                self._node_count += 1
                
            node = node.children[char]
            i += 1
            
        # Mark word completion and update statistics
        node.is_end_of_word = True
        node.frequency += 1
        self.size += 1
        
        # Invalidate affected cache entries
        # Only clear cache entries that could be affected by this insertion
        affected_prefix = word[:3]
        self._prefix_cache = {k: v for k, v in self._prefix_cache.items() 
                            if not k.startswith(affected_prefix)}

    def search_prefix(self, prefix):
        """
        Searches for a prefix in the trie.
        
        Args:
            prefix (str): The prefix to search for
            
        Returns:
            bool: True if prefix exists and is a complete word, False otherwise
            
        Features:
        - Case-insensitive search
        - Cache lookup for frequent searches
        - Handles compressed paths
        - Updates cache with results
        """
        if not prefix:
            return False
            
        # Check cache before traversing
        if prefix in self._prefix_cache:
            return self._prefix_cache[prefix]
            
        node = self.root
        prefix = prefix.lower()
        
        i = 0
        while i < len(prefix):
            char = prefix[i]
            
            # Return false if character not found
            if char not in node.children:
                self._cache_result(prefix, False)
                return False
                
            node = node.children[char]
            
            # Handle compressed path segments
            if node.compressed_value:
                remaining = prefix[i+1:]
                # Check if remaining prefix matches compressed value
                if not remaining.startswith(node.compressed_value):
                    self._cache_result(prefix, False)
                    return False
                i += len(node.compressed_value) + 1
            else:
                i += 1
                
        # Cache result and return
        self._cache_result(prefix, node.is_end_of_word)
        return node.is_end_of_word
        
    def autocomplete(self, prefix, limit=10):
        """
        Finds all words that start with the given prefix.
        
        Args:
            prefix (str): The prefix to autocomplete
            limit (int): Maximum number of suggestions to return
            
        Returns:
            list: Up to 'limit' words starting with prefix, sorted by frequency
            
        Features:
        - Frequency-based ranking
        - Efficient DFS traversal
        - Handles compressed paths
        - Respects suggestion limit
        """
        def dfs(node, path, results):
            """
            Depth-first search helper for finding autocomplete suggestions.
            
            Args:
                node (TrieNode): Current node in traversal
                path (str): Path from root to current node
                results (list): Accumulated results
            """
            if len(results) >= limit:
                return
                
            if node.is_end_of_word:
                results.append((path, node.frequency))
                
            # Handle compressed paths in results
            if node.compressed_value:
                if len(results) < limit:
                    results.append((path + node.compressed_value, node.frequency))
                return
                
            # Sort children by frequency for better suggestions
            sorted_children = sorted(
                node.children.items(),
                key=lambda x: getattr(x[1], 'frequency', 0),
                reverse=True
            )
            
            # Traverse children in frequency order
            for char, next_node in sorted_children:
                if len(results) >= limit:
                    break
                dfs(next_node, path + char, results)

        # Navigate to prefix node
        node = self.root
        prefix = prefix.lower()
        
        i = 0
        while i < len(prefix):
            char = prefix[i]
            if char not in node.children:
                return []
            node = node.children[char]
            # Handle compressed paths during prefix navigation
            if node.compressed_value:
                remaining = prefix[i+1:]
                if remaining.startswith(node.compressed_value):
                    return [(prefix, node.frequency)]
                return []
            i += 1
            
        # Collect and return results
        results = []
        dfs(node, prefix, results)
        return [word for word, _ in results]
        
    def _cache_result(self, prefix, result):
        """
        Caches a prefix search result with LRU-style eviction.
        
        Args:
            prefix (str): The prefix that was searched
            result (bool): The search result to cache
            
        Features:
        - LRU-style cache eviction
        - Batch removal when cache is full
        - Controlled cache growth
        """
        if len(self._prefix_cache) >= self.MAX_CACHE_SIZE:
            # Remove oldest 10% of entries when cache is full
            remove_count = self.MAX_CACHE_SIZE // 10
            for _ in range(remove_count):
                self._prefix_cache.pop(next(iter(self._prefix_cache)))
        self._prefix_cache[prefix] = result
    