import pandas as pd
import ast
import asyncio

from sentence_transformers import SentenceTransformer

from ragas import EvaluationDataset, evaluate
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings.base import BaseRagasEmbeddings
from ragas.run_config import RunConfig

from langchain_ollama import ChatOllama


# ============================================================
# 1. LOAD YOUR EXISTING EMBEDDING MODEL
# ============================================================

model = SentenceTransformer("all-MiniLM-L6-v2")


# ============================================================
# 2. CONNECT YOUR EMBEDDING MODEL TO RAGAS
# ============================================================

class MyEmbeddings(BaseRagasEmbeddings):

    def embed_query(self, text):
        embedding = model.encode(text)
        return embedding.tolist()

    def embed_documents(self, texts):
        embeddings = model.encode(texts)
        return embeddings.tolist()

    async def aembed_query(self, text):
        return self.embed_query(text)

    async def aembed_documents(self, texts):
        return self.embed_documents(texts)


# ============================================================
# 3. LOAD EVALUATION DATA
# ============================================================

df = pd.read_csv("Evaluation/results.csv")

print(f"Loaded {len(df)} evaluation questions")


# Convert context column from string → list
df["context"] = df["context"].apply(ast.literal_eval)


# ============================================================
# 4. CREATE RAGAS DATASET
# ============================================================

samples = []

for _, row in df.iterrows():

    sample = SingleTurnSample(
        user_input=row["question"],
        response=row["bot_answer"],
        retrieved_contexts=row["context"],
        reference=row["reference_answer"]
    )

    samples.append(sample)


evaluation_dataset = EvaluationDataset(
    samples=samples
)

print("RAGAS evaluation dataset created")


# ============================================================
# 5. LOCAL OLLAMA LLM
# ============================================================

ollama_llm = ChatOllama(
    model="llama3.1:8b",
    temperature=0,
    num_predict=512
)

evaluator_llm = LangchainLLMWrapper(
    ollama_llm
)

print("Ollama evaluator loaded")


# ============================================================
# 6. LOCAL EMBEDDINGS
# ============================================================

evaluator_embeddings = MyEmbeddings()

print("Existing all-MiniLM-L6-v2 embedding model connected")


# ============================================================
# 7. RAGAS METRICS
# ============================================================

metrics = [
    Faithfulness(llm=evaluator_llm),
    AnswerRelevancy(
        llm=evaluator_llm,
        embeddings=evaluator_embeddings
    ),
    ContextPrecision(
        llm=evaluator_llm
    ),
    ContextRecall(
        llm=evaluator_llm
    )
]


# ============================================================
# 8. RUN CONFIG
# ============================================================

run_config = RunConfig(
    timeout=300,
    max_workers=1,
    max_retries=3
)


# ============================================================
# 9. RUN EVALUATION
# ============================================================

print()
print("===================================")
print("STARTING RAGAS EVALUATION")
print("===================================")
print()

print("LLM       : Ollama llama3.2:3b")
print("Embedding : all-MiniLM-L6-v2")
print("Metrics   : 4")
print()

result = evaluate(
    dataset=evaluation_dataset,
    metrics=metrics,
    llm=evaluator_llm,
    embeddings=evaluator_embeddings,
    run_config=run_config,
    show_progress=True
)


# ============================================================
# 10. CONVERT RESULTS TO DATAFRAME
# ============================================================

result_df = result.to_pandas()


# ============================================================
# 11. DISPLAY RESULTS
# ============================================================

print()
print("===================================")
print("RAGAS RESULTS")
print("===================================")

print(
    result_df[
        [
            "user_input",
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall"
        ]
    ]
)


# ============================================================
# 12. SUMMARY
# ============================================================

print()
print("===================================")
print("EVALUATION SUMMARY")
print("===================================")

metric_names = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall"
]

for metric in metric_names:

    valid = result_df[metric].notna().sum()
    nan = result_df[metric].isna().sum()
    average = result_df[metric].mean()

    print()
    print(metric)

    print(f"Average score : {average:.4f}")
    print(f"Valid scores  : {valid}")
    print(f"NaN scores    : {nan}")


# ============================================================
# 13. SAVE RESULTS
# ============================================================

output_file = "Evaluation/ragas_results.csv"

result_df.to_csv(
    output_file,
    index=False
)

print()
print(f"Results saved to: {output_file}")