# Comparative Review #2: Comparison of 3 Entities (Tavily)

This project demonstrates a backend application that compares multiple entities (e.g., vector databases) using LangGraph, LangChain, and the Tavily search API.  
It generates comparison criteria, researches each entity against each criterion, builds a Markdown table, and produces a recommendation.

## Features

- **Dynamic Criteria Generation** – LLM generates 3‑5 relevant comparison criteria.
- **Iterative Research** – For every entity‑criterion pair, Tavily is queried and a short note is stored.
- **Markdown Table** – Findings are assembled into a readable table.
- **Recommendation** – LLM produces a concise verdict.
- **CLI Demo** – Run the script with default or custom entities.

## Prerequisites

- Python 3.10+
- An OpenAI API key (for the LLM)
- A Tavily API key

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/compare_entities.git
cd compare_entities

# Create a virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Configuration

Create a `.env` file in the project root (copy from `.env.example`):

```bash
cp .env.example .env
```

Edit `.env` and set your keys:

```dotenv
# .env
OPENAI_API_KEY=your_openai_api_key
TAVILY_API_KEY=your_tavily_api_key
```

## Usage

Run the demo with the default entities:

```bash
python main.py
```

You can provide custom entities and a query:

```bash
python main.py --entities "Chroma, FAISS, Qdrant, Milvus" \
               --query "Compare vector databases for real‑time recommendation systems"
```

The script will output:

1. A Markdown table summarizing the findings for each criterion.
2. A short recommendation on which entity best fits the use case.

## Project Structure

```
compare_entities/
├── main.py          # Entry point and graph definition
├── README.md        # Documentation
├── .env.example     # Sample environment file
├── requirements.txt # Python dependencies
```

## Dependencies

- `langgraph` – Graph orchestration
- `langchain-openai` – OpenAI LLM integration
- `langchain-tavily` – Tavily API wrapper
- `tavily-python` – Tavily client
- `python-dotenv` – Environment variable loading

Install them via `pip install -r requirements.txt`.

## License

MIT License
