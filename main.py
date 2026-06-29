import os
import json
import argparse
from typing import TypedDict, List, Dict, Any, Tuple

from langgraph import StateGraph, END
from langchain_openai import ChatOpenAI
from tavily import TavilyClient
from dotenv import load_dotenv

load_dotenv()

class CompareState(TypedDict):
    entities: List[str]
    criteria: List[str]
    findings: List[Dict[str, Any]]
    final_table: str
    verdict: str
    remaining_pairs: List[Tuple[str, str]]
    query: str

# Initialize LLM
llm = ChatOpenAI()

def _response_content(response) -> str:
    return getattr(response, "content", str(response))

def plan_criteria(state: CompareState) -> CompareState:
    prompt = (
        f"Generate 3-5 comparison criteria for evaluating the following entities: "
        f"{', '.join(state['entities'])}. "
        f"The criteria should be relevant to the use case: {state.get('query', 'General comparison')}. "
        f"Return a JSON array of strings."
    )
    response = llm.invoke(prompt)
    content = _response_content(response)
    try:
        criteria = json.loads(content)
    except Exception:
        criteria = [line.strip() for line in content.splitlines() if line.strip()]
    state["criteria"] = criteria
    return state

def tavily_search(query: str) -> str:
    client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    try:
        results = client.search(query, max_results=1)
        if results:
            result = results[0]
            return result.get("content") or result.get("snippet") or result.get("title") or "No summary available."
        return "No results found."
    except Exception as e:
        return f"Error: {str(e)}"

def research_entity(state: CompareState) -> CompareState:
    if not state.get("remaining_pairs"):
        state["remaining_pairs"] = [(e, c) for e in state["entities"] for c in state["criteria"]]
    if not state["remaining_pairs"]:
        return state
    entity, criterion = state["remaining_pairs"].pop(0)
    query = f"{entity} {criterion}"
    note = tavily_search(query)
    state["findings"].append({"entity": entity, "criterion": criterion, "note": note})
    return state

def build_table(state: CompareState) -> CompareState:
    entities = state["entities"]
    criteria = state["criteria"]
    findings = state["findings"]
    header = "| Criterion | " + " | ".join(entities) + " |\n"
    separator = "|" + "--------|" * (len(entities) + 1) + "\n"
    rows = ""
    for criterion in criteria:
        row = f"| {criterion} | "
        for entity in entities:
            note = next(
                (f["note"] for f in findings if f["entity"] == entity and f["criterion"] == criterion),
                "N/A",
            )
            row += f"{note} | "
        rows += row + "\n"
    table = header + separator + rows
    state["final_table"] = table
    return state

def verdict(state: CompareState) -> CompareState:
    prompt = (
        f"Based on the following comparison table, recommend the best entity for the given use case:\n\n"
        f"{state['final_table']}\n\n"
        f"Provide 2-4 sentences with a clear recommendation."
    )
    response = llm.invoke(prompt)
    state["verdict"] = _response_content(response).strip()
    return state

def build_graph() -> StateGraph:
    graph = StateGraph(CompareState)
    graph.add_node("plan_criteria", plan_criteria)
    graph.add_node("research_entity", research_entity)
    graph.add_node("build_table", build_table)
    graph.add_node("verdict", verdict)

    graph.set_entry_point("plan_criteria")
    graph.add_conditional_edges(
        "plan_criteria",
        lambda _: "research_entity",
        {"research_entity": "research_entity"},
    )
    graph.add_conditional_edges(
        "research_entity",
        lambda state: "research_entity" if state["remaining_pairs"] else "build_table",
        {"research_entity": "research_entity", "build_table": "build_table"},
    )
    graph.add_edge("build_table", "verdict")
    graph.add_edge("verdict", END)

    return graph

def main():
    parser = argparse.ArgumentParser(description="Compare entities using LangGraph and Tavily.")
    parser.add_argument(
        "--entities",
        type=str,
        default="Chroma, FAISS, Qdrant",
        help="Comma-separated list of entities to compare.",
    )
    parser.add_argument(
        "--query",
        type=str,
        default="Сравни 3 векторные БД: Chroma, FAISS, Qdrant",
        help="Use case query for comparison.",
    )
    args = parser.parse_args()

    entities = [e.strip() for e in args.entities.split(",") if e.strip()]
    if not entities:
        print("No entities provided.")
        return

    initial_state: CompareState = {
        "entities": entities,
        "criteria": [],
        "findings": [],
        "final_table": "",
        "verdict": "",
        "remaining_pairs": [],
        "query": args.query,
    }

    graph = build_graph()
    compiled = graph.compile()
    final_state = compiled.invoke(initial_state)

    print("\n=== Comparison Table ===\n")
    print(final_state["final_table"])
    print("\n=== Verdict ===\n")
    print(final_state["verdict"])

if __name__ == "__main__":
    main()
