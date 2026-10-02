
from Src.pdf_loader import Loadpdf  
from Src.Chunker import create_chunks
from Src.Embeddings import get_embedding
from Src.VectorDB import store_embeddings, get_collection
from pdf_loader import Loadpdf




# Extract PDF
pdf_path = r"D:\RAG_Project\Industrial_RAG_Assistant\Data\ElectricalEngineering.pdf"

pages = Loadpdf(pdf_path)   # load pdf

print(f"Total number of pages: {len(pages)}")

# print("Content of the first page:")
# print(pages[0][:1000])


# Create chunks
chunks = create_chunks(pages)

print(f"Total chunks: {len(chunks)}")


# Show first chunk
# print("\n--- First Chunk ---")
# print(chunks[0])

# Create Embeddings
embeddings = [get_embedding(chunk) for chunk in chunks]

print(f"Total embeddings: {len(embeddings)}")
print(f"Embedding dimension: {len(embeddings[0])}")


# store embeddings in ChromaDB
store_embeddings(chunks, embeddings)