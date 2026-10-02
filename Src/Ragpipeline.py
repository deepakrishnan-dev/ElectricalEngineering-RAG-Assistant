from Src.Retriever import get_similar_chunks
from Src.Prompt import create_prompt
from Src.LLM import generate_response


class RAGPipeline:

    def __init__(self, top_k=3):
        self.top_k = top_k

    def answer(self, question):

        retrieved_chunks = get_similar_chunks(
            question,
            top_k=self.top_k
        )

        context = "\n\n".join(
            retrieved_chunks["documents"][0]
        )

        prompt = create_prompt(
            question,
            context
        )

        answer = generate_response(prompt)

        return answer