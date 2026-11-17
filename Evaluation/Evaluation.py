import pandas as pd
from Src.Ragpipeline import RAGPipeline

# Load RAG pipeline
rag_pipeline = RAGPipeline(top_k=3)

# Load evaluation questions
df = pd.read_csv("Evaluation/evaluation_questions.csv")

# Create columns before the loop
df["bot_answer"] = ""
df["context"] = None

# Generate answers and retrieve contexts
for index, row in df.iterrows():

    question = row["question"]

    print(f"processing question {index + 1}")

    # Get answer and retrieved contexts
    answer, context = rag_pipeline.answer(question)

    # Save answer
    df.loc[index, "bot_answer"] = answer

    # Save retrieved chunks
    df.at[index, "context"] = context

# Save results
df.to_csv("Evaluation/results.csv", index=False)

print("all answers and contexts saved successfully")