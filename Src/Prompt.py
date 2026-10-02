

def create_prompt(query,context):

    prompt = f"""
    You are an AI assistant for answering questions related to Electrical Engineering. 
    Use the following context to answer the question. 

    Context:
    {context}

    Question: {query}
    Instructions:
            - Answer only using the provided context.
            - Do not make up information.
            - If the answer is not available in the context, say that the information is not available.
    """
    return prompt