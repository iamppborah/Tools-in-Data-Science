# /// script
# requires-python = "===3.12.7"
# dependencies = [
#   "pandas==2.2.3",
#   "seaborn==0.13.2",
#   "matplotlib==3.10.0",
#   "httpx==0.28.1",
#   "tenacity==9.0.0",
#   "chardet==5.2.0"
# ]
# ///


import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import httpx
from tenacity import retry, stop_after_attempt, wait_fixed
import chardet

# Set up environment variables for API access
AI_PROXY = os.environ.get("AI_PROXY")

# Define the LLM query function with retry mechanism using httpx to match the curl example
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def query_llm(prompt):
    headers = {"Authorization": f"Bearer {AI_PROXY}", "Content-Type": "application/json"}
    data = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    }
    # Specify a longer timeout (e.g., 30 seconds)
    timeout = httpx.Timeout(30.0, connect=60.0)
    response = httpx.post("https://api.openai.com/v1/chat/completions", json=data, headers=headers, timeout=timeout)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

# Function to detect file encoding
def detect_file_encoding(file_path):
    with open(file_path, 'rb') as file:
        result = chardet.detect(file.read())
    return result['encoding']

# Function to read CSV file with detected encoding
def read_csv_file(filename):
    encoding = detect_file_encoding(filename)
    # Specify dtype for columns that should be treated as strings to avoid conversion errors
    dtype_spec = {'Country name': str}
    return pd.read_csv(filename, encoding=encoding, dtype=dtype_spec)

# Function to perform generic analysis on the dataframe
def analyze_data(df):
    # Select only numeric columns for correlation matrix calculation
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    summary_stats = numeric_df.describe()
    missing_values = numeric_df.isnull().sum()
    correlation_matrix = numeric_df.corr()
    return summary_stats, missing_values, correlation_matrix

# Function to visualize correlation matrix
def visualize_correlation(correlation_matrix):
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
    plt.title("Correlation Matrix")
    plt.savefig("correlation_matrix.png")

# Main function to process the CSV file
def process_csv_file(filename):
    df = read_csv_file(filename)
    summary_stats, missing_values, correlation_matrix = analyze_data(df)

    # Use LLM to get insights from summary statistics and missing values
    insights_prompt = f"Provide insights based on this summary: {summary_stats.to_string()} and missing values: {missing_values.to_string()}"
    insights = query_llm(insights_prompt)

    # Visualize the correlation matrix
    visualize_correlation(correlation_matrix)

    # Use LLM to narrate a story from the analysis
    narrative_prompt = f"Narrate a story from the analysis and insights: {insights}"
    narrative = query_llm(narrative_prompt)

    # Save the narrative and insights to README.md
    with open("README.md", "w") as f:
        f.write(narrative)
        f.write("\n\n## Correlation Matrix\n")
        f.write("![Correlation Matrix](correlation_matrix.png)")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: uv run autolysis.py <path_to_csv_file>")
        sys.exit(1)

    csv_file_path = sys.argv[1]
    process_csv_file(csv_file_path)
