import chromadb
from sentence_transformers import SentenceTransformer

# 1. Load the exact same model you used in db.py
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. Connect to the database you ALREADY built
# (Make sure you ran db.py first so the 'chroma_db_storage' folder exists!)
chroma_client = chromadb.PersistentClient(path="./chroma_db_storage")
collection = chroma_client.get_collection(name="data")

# Add 'subject' as an argument. Default it to "All" just in case.
def get_relevant_course_context(user_query: str, subject: str = "All", max_distance: float = 1.2) -> str:
    
    query_vector = embedding_model.encode(user_query).tolist()
    
    # Set up the basic search parameters
    query_params = {
        "query_embeddings": [query_vector],
        "n_results": 3
    }
    
    # If a specific subject is chosen, tell ChromaDB to filter for it
    if subject != "All":
        query_params["where"] = {"subject": subject}
        
    # Execute the search with our parameters
    results = collection.query(**query_params)
    
    valid_chunks = []
    
    # ⚠️ THIS IS THE FINAL FIXED LOGIC
    # We must append to extract the inner lists!
    if results['documents'] and len(results['documents']) > 0:
        if results['documents'] and results['documents'][0]:
            docs_list = results['documents'][0]        # ✅ unwrap
            distances_list = results['distances'][0]   # ✅ unwrap
            for i in range(len(docs_list)):
                doc = docs_list[i]
                distance = distances_list[i]  # ✅ now it's a float

                if distance <= max_distance:
                    valid_chunks.append(doc)
                
    if not valid_chunks:
        return "INSUFFICIENT_CONTEXT"
        
    return "\n\n---\n\n".join(valid_chunks)


# --- QUICK TEST ---
if __name__ == "__main__":
    subject = "OS"  # ✅ define it
    answer = get_relevant_course_context("What is a photosynthesis?", subject=subject)
    print("Testing " + subject + " subject filter...")
    print(answer)