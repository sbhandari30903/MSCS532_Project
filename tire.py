class TrieNode:
    def __init__(self):
        # Initialize an empty dictionary 
        self.children = {}
        self.is_end_of_word = False

class Trie:
    def __init__(self):
        # Create the root node
        self.root = TrieNode()

    def insert(self, word):
        # Start at the root node
        node = self.root
        for char in word.lower():
            # Add a new node for the character if it doesn't exist
            if char not in node.children:
                node.children[char] = TrieNode()
            # Move to the next node
            node = node.children[char]
        # Mark the end of the word
        node.is_end_of_word = True

    def search_prefix(self, prefix):
        # Start at the root node
        node = self.root
        for char in prefix.lower():
            # If the character is not found
            if char not in node.children:
                return False
            # Move to the next node
            node = node.children[char]
        # if the prefix exists in the trie
        return True
    
    def autocomplete(self, prefix):
        def dfs(node, path, results):
            # if not in the list
            if node.is_end_of_word:
                results.append(path)
            for char, next_node in node.children.items():
                dfs(next_node, path + char, results)

        node = self.root
        for char in prefix.lower():
            if char not in node.children:
                return []
            node = node.children[char]
        # matching results
        results = []
        dfs(node, prefix.lower(), results)
        return results
    