#initializing
import os
import PyPDF2
import chromadb
from sentence_transformers import SentenceTransformer

print("Waking up the embedding model...")
# This is the brain that turns English words into math vectors
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

print("Opening the vault (ChromaDB)...")
# This creates a literal folder on your computer to save the database permanently
chroma_client = chromadb.PersistentClient(path="./chroma_db_storage")

# Think of a 'collection' like a specific table in an SQL database
collection = chroma_client.get_or_create_collection(name="data")


#pdf parsing
def extract_text_from_pdf(pdf_path):
    """Opens a PDF and rips out all the text page by page."""
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        # Loop through every single page
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text() + "\n"
    return text


#chunker
def chunk_text(text, words_per_chunk=150, overlap=30):
    """
    Chops the massive wall of text into small blocks.
    The overlap makes sure we don't accidentally slice a sentence in half.
    """
    words = text.split()
    chunks = []
    
    # Step through the words, stepping back slightly each time for the overlap
    for i in range(0, len(words), words_per_chunk - overlap):
        chunk = " ".join(words[i:i + words_per_chunk])
        chunks.append(chunk)
        
    return chunks



def build_database():
    data_folder = "./data"
    
    # Safety check: make sure the folder exists
    if not os.path.exists(data_folder):
        print(f"Bro, I can't find the '{data_folder}' folder. Create it and add PDFs.")
        return

    # Use os.walk to go into your subfolders (OS, COA, etc.)
    for root, dirs, files in os.walk(data_folder):
        
        # Extract the subject name from the folder name
        # If root is "./data/OS", subject_name becomes "OS"
        subject_name = os.path.basename(root)
        
        # If there are loose PDFs in the main data folder, label them 'General'
        if subject_name == "data":
            subject_name = "General"

        for filename in files:
            if filename.endswith(".pdf"):
                print(f"\nProcessing [{subject_name}]: {filename}...")
                pdf_path = os.path.join(root, filename)
                
                # Step A: Rip the text
                raw_text = extract_text_from_pdf(pdf_path)
                
                # Step B: Chop it into pieces
                chunks = chunk_text(raw_text)
                
                # Step C: Turn into math and save to database
                for i, chunk in enumerate(chunks):
                    chunk_id = f"{filename}_chunk_{i}"
                    vector = embedding_model.encode(chunk).tolist()
                    
                    # ⚠️ CRITICAL UPDATE: We are adding "subject" to the metadata!
                    collection.upsert(
                        ids=[chunk_id],
                        embeddings=[vector],
                        documents=[chunk],
                        metadatas=[{"source": filename, "subject": subject_name}]
                    )
                
    print("\n✅ Database build complete! The RAG is loaded and ready.")

# This tells Python to actually run the master switch when you execute the file
if __name__ == "__main__":
    build_database()