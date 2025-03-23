import os
import re
import json
import nltk
from nltk.stem import PorterStemmer
from collections import defaultdict
import datetime

class InformationRetrievalSystem:
    def __init__(self, docs_folder, stop_words_file):
        """
        Args:
            docs_folder (str): Path to folder containing document abstracts
            stop_words_file (str): Path to file containing stop words
        """
        self.docs_folder = docs_folder
        self.stop_words_file = stop_words_file
        self.stop_words = self._load_stop_words()
        self.stemmer = PorterStemmer()
        self.inverted_index = defaultdict(list)  # term -> [doc_ids]
        self.positional_index = defaultdict(lambda: defaultdict(list))  # term -> {doc_id -> [positions]}
        self.doc_ids = {} # doc_id -> filename
        self._build_indexes()

    #load Stop words from file
    def _load_stop_words(self):
        stop_words = set()
        try:
            with open(self.stop_words_file, 'r') as f:
                for line in f:
                    word = line.strip()
                    if word:
                        stop_words.add(word)
            return stop_words
        except FileNotFoundError:
            print(f"Warning: Stop words file {self.stop_words_file} not found.")
            return set()
    
    # removing stop words and stemming
    # tokenizing the text
    def _preprocess_text(self, text):
        tokens = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        preprocessed_tokens = []
        for pos, token in enumerate(tokens):
            if token not in self.stop_words:
                stemmed_token = self.stemmer.stem(token)
                preprocessed_tokens.append((stemmed_token, pos))
        return preprocessed_tokens
    
    def _build_indexes(self):
        print("Building indexes...")
        try:
            if os.path.exists('indexes.json'):
                self._load_indexes()
                print("Loaded existing indexes.")
                return
        except Exception as e:
            print(f"Error loading saved indexes: {e}")
            print("Building indexes from scratch...")

        processed_docs = 0
        failed_docs = []
        
        for doc_id, filename in enumerate(os.listdir(self.docs_folder)):
            if not filename.endswith('.txt'):
                continue
                
            file_path = os.path.join(self.docs_folder, filename)
            
            try:
                content = self._read_document(file_path)
                
                # Process document
                self.doc_ids[doc_id] = filename
                preprocessed_tokens = self._preprocess_text(content)
                
                # Update indexes
                terms_in_doc = set()
                for term, position in preprocessed_tokens:
                    terms_in_doc.add(term)
                    self.positional_index[term][doc_id].append(position)
                
                for term in terms_in_doc:
                    self.inverted_index[term].append(doc_id)
                    
                processed_docs += 1
                
            except Exception as e:
                failed_docs.append((filename, str(e)))
        
        # Log processing results
        print(f"\nProcessing Summary:")
        print(f"Successfully processed: {processed_docs} documents")
        if failed_docs:
            print("\nFailed documents:")
            for doc, error in failed_docs:
                print(f"- {doc}: {error}")
        
        self._save_indexes()
    
    def _save_indexes(self):
        index_data = {
            'metadata': {
                'total_documents': len(self.doc_ids),
                'total_terms': len(self.inverted_index),
                'created_at': str(datetime.datetime.now())
            },
            'document_mapping': {
                str(doc_id): {
                    'filename': filename,
                    'path': os.path.join(self.docs_folder, filename)
                }
                for doc_id, filename in self.doc_ids.items()
            },
            'inverted_index': {
                term: {
                    'document_frequency': len(postings),
                    'postings': postings
                }
                for term, postings in self.inverted_index.items()
            },
            'positional_index': {
                term: {
                    str(doc_id): positions
                    for doc_id, positions in postings.items()
                }
                for term, postings in self.positional_index.items()
            }
        }
        
        try:
            with open('indexes.json', 'w', encoding='utf-8') as f:
                json.dump(index_data, f, indent=2, ensure_ascii=False)
            print("Indexes saved successfully")
        except Exception as e:
            print(f"Error saving indexes: {e}")
    
    def _load_indexes(self):
        with open('indexes.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Load document mapping
        self.doc_ids = {
            int(doc_id): info['filename']
            for doc_id, info in data['document_mapping'].items()
        }
        
        # Load inverted index
        self.inverted_index = defaultdict(list)
        for term, info in data['inverted_index'].items():
            self.inverted_index[term] = info['postings']
        
        # Load positional index
        self.positional_index = defaultdict(lambda: defaultdict(list))
        for term, postings in data['positional_index'].items():
            for doc_id, positions in postings.items():
                self.positional_index[term][int(doc_id)] = positions
    
    def _read_document(self, file_path):
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1'] # used all possible encodings for reading the file

        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        
        # If all encodings fail, try reading in binary mode and decode with errors ignored
        with open(file_path, 'rb') as f:
            return f.read().decode('utf-8', errors='ignore')
    
    # main function to process the query
    def process_query(self, query):

        # Check for proximity query
        if '/' in query:
            return self._process_proximity_query(query)
        
        # Check for Boolean operators
        if ' AND ' in query or ' OR ' in query or 'NOT ' in query:
            return self._process_boolean_query(query)
        
        # Simple term query
        term = query.strip().lower()
        stemmed_term = self.stemmer.stem(term)
        
        # Return document IDs containing the term
        return sorted(self.inverted_index.get(stemmed_term, []))
    
    # for processing the boolean query
    def _process_boolean_query(self, query):
        # Handle queries with brackets
        if '(' in query:
            return self._process_bracketed_query(query)
            
        tokens = re.split(r'\s+(AND|OR|NOT)\s+', query)        
        if len(tokens) == 1:
            # Single term query
            term = tokens[0].lower()
            stemmed_term = self.stemmer.stem(term)
            return sorted(self.inverted_index.get(stemmed_term, []))
        
        # Initialize result with the first term
        term = tokens[0].lower()
        stemmed_term = self.stemmer.stem(term)
        result = set(self.inverted_index.get(stemmed_term, []))
        
        i = 1
        while i < len(tokens):
            operator = tokens[i]
            term = tokens[i + 1].lower() if i + 1 < len(tokens) else ""
            
            if not term:
                break
                
            stemmed_term = self.stemmer.stem(term)
            term_docs = set(self.inverted_index.get(stemmed_term, []))
            
            if operator == 'AND':
                result = result.intersection(term_docs)
            elif operator == 'OR':
                result = result.union(term_docs)
            elif operator == 'NOT':
                result = result.difference(term_docs)
            
            i += 2
        
        return sorted(list(result))

    def _process_bracketed_query(self, query):
        """
        Process boolean queries containing brackets for nested expressions.
        Example: "computer AND science NOT (Times AND Series)"
        """
        if '(' not in query:
            return self._process_boolean_query(query)
        
        stack = []
        start = -1
        
        for i, char in enumerate(query):
            if char == '(':
                stack.append(i)
            elif char == ')' and stack:
                start = stack.pop()
                end = i
                break
        
        if start != -1:
            inner_expr = query[start + 1:end].strip()
            inner_result = self._process_boolean_query(inner_expr)
            
            placeholder = f"RESULT_{hash(str(inner_result))}"
            
            new_query = query[:start] + placeholder + query[end + 1:]
            
            final_result = self._process_bracketed_query(new_query)
            
            # If the placeholder is directly in the result, we need to replace it
            if isinstance(final_result, list) and len(final_result) == 1 and str(final_result[0]).startswith("RESULT_"):
                return inner_result
            
            # Process the query again with operators
            tokens = re.split(r'\s+(AND|OR|NOT)\s+', new_query)
            
            # Initialize result based on the first token
            if tokens[0] == placeholder:
                result = set(inner_result)
            else:
                term = tokens[0].lower()
                stemmed_term = self.stemmer.stem(term)
                result = set(self.inverted_index.get(stemmed_term, []))
            
            i = 1
            while i < len(tokens):
                operator = tokens[i]
                term = tokens[i + 1] if i + 1 < len(tokens) else ""
                
                if not term:
                    break
                
                # Check if the term is our placeholder
                if term == placeholder:
                    term_docs = set(inner_result)
                else:
                    stemmed_term = self.stemmer.stem(term.lower())
                    term_docs = set(self.inverted_index.get(stemmed_term, []))
                
                if operator == 'AND':
                    result = result.intersection(term_docs)
                elif operator == 'OR':
                    result = result.union(term_docs)
                elif operator == 'NOT':
                    result = result.difference(term_docs)
                
                i += 2
            
            return sorted(list(result))
        
        return self._process_boolean_query(query)
    
    # for processing the proximity query
    def _process_proximity_query(self, query):
        # Parse the proximity query
        match = re.match(r'(.*?)\s+(/\s*(\d+))$', query)
        if not match:
            print("Invalid proximity query format. Use 'term1 term2 /k'")
            return []
        
        terms_part = match.group(1)
        k = int(match.group(3))
        
        # Extract terms
        terms = terms_part.lower().split()
        if len(terms) != 2:
            print("Proximity query supports only 2 terms. Use 'term1 term2 /k'")
            return []
        
        # Stem the terms
        stemmed_terms = [self.stemmer.stem(term) for term in terms]
        
        # Find documents containing both terms
        docs_term1 = set(self.inverted_index.get(stemmed_terms[0], []))
        docs_term2 = set(self.inverted_index.get(stemmed_terms[1], []))
        common_docs = docs_term1.intersection(docs_term2)
        
        result_docs = []
        
        # Check proximity in each document
        for doc_id in common_docs:
            positions1 = self.positional_index[stemmed_terms[0]][doc_id]
            positions2 = self.positional_index[stemmed_terms[1]][doc_id]
            
            # Check if any positions are within k words of each other
            for pos1 in positions1:
                for pos2 in positions2:
                    if abs(pos1 - pos2) <= k:
                        result_docs.append(doc_id)
                        break
                else:
                    continue
                break
        
        return sorted(result_docs)
    
    def print_results(self, doc_ids):
        if not doc_ids:
            print("No matching documents found.")
            return
        
        file_numbers = []
        for doc_id in doc_ids:
            filename = self.doc_ids[doc_id]
            try:
                number = int(re.search(r'\d+', filename).group())
                file_numbers.append(number)
            except (AttributeError, ValueError):
                file_numbers.append(doc_id)
        
        result_str = ", ".join(map(str, sorted(file_numbers)))
        print(f"Result-Set: {result_str}")
        return result_str

def main():
    docs_folder = "Abstracts"  # Path to the folder containing abstract documents
    stop_words_file = "stop_words.txt"  # Path to the stop words file
    
    ir_system = InformationRetrievalSystem(docs_folder, stop_words_file)
    
    print("\nBoolean Information Retrieval System")
    print("===================================")
    print("Query syntax:")
    print("- Simple term query: term")
    print("- Boolean query: term1 AND term2, term1 OR term2, NOT term")
    print("- Complex Boolean query: term1 AND term2 AND term3")
    print("- Proximity query: term1 term2 /k (where k is the maximum distance)")
    print("- Enter 'exit' to quit")
    print("===================================\n")
    
    while True:
        query = input("Enter your query: ").strip()
        
        if query.lower() == 'exit':
            break
        
        if not query:
            continue
        
        doc_ids = ir_system.process_query(query)
        ir_system.print_results(doc_ids)
        print()

if __name__ == "__main__":
    # Check if NLTK stopwords are available, download if needed
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("Downloading NLTK resources...")
        nltk.download('punkt')
    
    main()