import nltk
from boolean_retrieval_model import InformationRetrievalSystem

def run_test_queries():
    docs_folder = "Abstracts"
    stop_words_file = "stop_words.txt"
    ir_system = InformationRetrievalSystem(docs_folder, stop_words_file)
    
    queries = [
        "image AND restoration",
        "deep AND learning",
        "autoencoders",
        "temporal AND deep AND learning",
        "time AND series",
        "time AND series AND classification",
        "time AND series OR classification",
        "pattern",
        "pattern AND clustering",
        "pattern AND clustering AND heart",
        "neural information /2",
        "feature track /5"
    ]
    
    print("\nRunning test queries...\n")
    for query in queries:
        print(f"Query: {query}")
        doc_ids = ir_system.process_query(query)
        ir_system.print_results(doc_ids)
        print()

if __name__ == "__main__":
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("Downloading NLTK resources...")
        nltk.download('punkt')
    run_test_queries()