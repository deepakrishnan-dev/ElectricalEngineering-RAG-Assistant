from Src.Embeddings import get_embedding
from Src.VectorDB import collection


def get_similar_chunks(query,top_k=5):
    qns_embedding = get_embedding(query)

    results = collection.query(
        query_embeddings=[qns_embedding],
        n_results=top_k
    )

    return results