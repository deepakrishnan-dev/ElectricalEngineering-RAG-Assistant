from Src.Ragpipeline import RAGPipeline


def main():

    rag = RAGPipeline(top_k=3)

    while True:

        query = input("\nEnter your query: ")

        if query.lower() == "exit":
            print("Goodbye!")
            break

        answer = rag.answer(query)

        print("\n--- Answer ---")
        print(answer)


if __name__ == "__main__":
    main()