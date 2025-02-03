class InvertedIndex:
    def __init__(self):
        # Initialize empty dictionary to store the inverted index
        self.index = {}

    def add_document(self, doc_id, content):
        # Split the document content into individual words
        words = content.lower().split()
        for word in words:
            # If the word is not in index, add it with an empty list
            if word not in self.index:
                self.index[word] = set()
            # Append the document ID to the list of occurrences for the word
            self.index[word].add(doc_id)

    def search(self, word):
        # Retrieve the list of document IDs for the given word, or an empty list if not found
        return self.index.get(word.lower(), set())
    
