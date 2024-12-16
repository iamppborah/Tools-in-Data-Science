# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas==1.4.1",
#   "seaborn==0.11.2",
#   "matplotlib==3.5.1",
#   "httpx==0.22.0",
#   "tenacity==8.0.1",  # Ensure this line is present and correctly formatted
# ]
# ///


import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import httpx
from tenacity import retry, stop_after_attempt, wait_fixed

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def query_llm(prompt):
    token = os.environ["AI_PROXY"]
    headers = {"Authorization": f"Bearer {token}"}
    response = httpx.post("https://api.openai.com/v1/completions",
                          json={"model": "gpt-4o-mini", "prompt": prompt, "max_tokens": 100},
                          headers=headers)
    response.raise_for_status()
    return response.json()["choices"][0]["text"]

def read_csv(filename):
    return pd.read_csv(filename)

def perform_generic_analysis(df):
    summary = df.describe(include='all')
    missing_values = df.isnull().sum()
    correlation_matrix = df.corr()
    return summary, missing_values, correlation_matrix

def visualize_correlation_matrix(correlation_matrix):
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f")
    plt.savefig("correlation_matrix.png")

def main(csv_filename):
    df = read_csv(csv_filename)
    summary, missing_values, correlation_matrix = perform_generic_analysis(df)

    # Interact with LLM for analysis suggestions
    prompt = f"Analyze this dataset: Summary: {summary.to_string()}, Missing values: {missing_values.to_string()}"
    analysis_suggestion = query_llm(prompt)

    # Visualize results
    visualize_correlation_matrix(correlation_matrix)

    # Narrate a story
    narrative_prompt = f"Narrate a story based on this analysis: {analysis_suggestion}"
    narrative = query_llm(narrative_prompt)

    # Write results to README.md
    with open("README.md", "w") as md_file:
        md_file.write(narrative)
        md_file.write("\n\n![Correlation Matrix](correlation_matrix.png)")

if __name__ == "__main__":
    import sys
    csv_filename = sys.argv[1]
    main(csv_filename)