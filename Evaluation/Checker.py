
import pandas as pd

df = pd.read_csv("Evaluation/ragas_faithfulness_results.csv")

# display the questions which get faithfullnes is Nan
nan_questions = df[df['faithfulness'].isna()]
