import chromadb

client = chromadb.PersistentClient(path="chroma_db")

# create and open a collection
collection = client.get_or_create_collection("electrical_engineering")

def store_embeddings(chunks,embeddings):

    id = [f"chunk_{i}" for i in range(len(chunks))]

    # store the chunks and embeddings in the collection
    collection.add(
        ids=id,
        documents=chunks,
        embeddings=embeddings
    )
    print(f"Stored {len(chunks)} chunks in ChromaDB")

def get_collection():

    data=collection.get(include=["documents","embeddings"])
    print(f"Total number of chunks in the collection: {len(data['ids'])}")
    print(f"First chunk: {data['documents'][0]}")
    print(f"First chunk embedding: {data['embeddings'][0]}")
