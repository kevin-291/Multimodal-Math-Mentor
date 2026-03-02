import os
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_community.utilities.wolfram_alpha import WolframAlphaAPIWrapper
from dotenv import load_dotenv
from .llm import llm
from utils.rag import RAG
from utils.memory_bank import search_memory
from .state import AgentState

load_dotenv() 

@tool
def calculator_tool(expression: str) -> str:
    """A tool that evaluates a mathematical expression and returns the result as a string.
    Args:
        expression: A string containing the mathematical expression to evaluate. For example, "2 + 2" or "integrate x^2 dx".
    Returns:
        a string containing the result of the calculation. For example, "4" or "x^3/3 + C".
    """
    try:
        app_id = os.getenv("WOLFRAM_ALPHA_APPID")
        wolfram = WolframAlphaAPIWrapper(wolfram_alpha_appid=app_id)
        return wolfram.run(expression)
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"

model = llm()
KNOWLEDGE_BASE_DIR = "knowledge_base/" 
rag_system = RAG(KNOWLEDGE_BASE_DIR)

SYSTEM_PROMPT = """
You are a helpful assistant that solves math problems.
You have access to a RAG system, a calculator tool, and a memory bank of past verified solutions.
If past solutions are provided, reuse their patterns and correct any known OCR/Audio typos mentioned in them.
"""

agent = create_agent(
    model=model,
    tools=[calculator_tool],
    system_prompt=SYSTEM_PROMPT
)

def solver_agent_node(state: AgentState):
    question = state.get("text_input", "").replace("\n", " ").strip()

    retrieved_docs = rag_system.retrieve(question, top_k=1)
    context = "\n\n".join(retrieved_docs)

    past_items = search_memory(question, limit=1)
    past_patterns = None
    if past_items:
        past_patterns = "\n\n".join([
            f"Past Input: {item['original_input']}\nPast Verified Solution: {item['final_answer']}" 
            for item in past_items
        ])

    user_prompt = f"""
    Solve the following problem.
    
    Problem: {question}
    
    Relevant Knowledge Base Information:
    {context}
    
    Similar Past Verified Solutions (Use these to guide your steps and correct typos):
    {past_patterns if past_patterns else "No similar past solutions found."}
    
    Use the provided information and your calculator tool to determine the final solution.
    """

    response = agent.invoke({
        "messages": [HumanMessage(content=user_prompt)]
    })

    answer = response["messages"][-1].content

    return {
        **state,
        "retrieved_context": context,
        "past_similar_problems": past_patterns,
        "solution": answer  
    }