from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
from utils.memory_bank import save_memory
from .llm import llm
from .state import AgentState

model = llm()

class VerifierOutput(BaseModel):
    correct: bool = Field(..., description="True if the mathematical solution is fully correct and logically sound. False otherwise.")
    feedback: str = Field(..., description="Detailed explanation of any mathematical flaws, missing domain checks, or 'All good.' if perfectly correct.")
    needs_hitl: bool = Field(..., description="True if the problem is highly ambiguous, the steps are nonsensical, or you are unsure and need human review. False otherwise.")

SYSTEM_PROMPT = """
You are a rigorous Mathematics Verifier for JEE-level problems.
Review the provided problem along with the topic, variables, constraints, and the proposed solution. 
Critique based on mathematical correctness, units, domain constraints (e.g., division by zero), and edge cases.
"""

verifier_agent = create_agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
    response_format=VerifierOutput
)

def verifier_agent_node(state: AgentState):
    user_input = f"""
    Problem: {state.get("text_input", "")}\n
    Topic: {state.get("parsed_topic", "Unknown")}\n
    Variables: {', '.join(state.get("parsed_variables", []))}\n
    Constraints: {', '.join(state.get("parsed_constraints", []))}\n
    Proposed Solution: {state.get("solution", "")}
    """
    
    response = verifier_agent.invoke({
        "messages": [HumanMessage(content=user_input)]
    })
    

    result = response["structured_response"]

    save_memory(
        original_input=state.get("text_input", ""),
        retrieved_context=state.get("retrieved_context", ""),
        final_answer=state.get("solution", ""),
        verifier_outcome=result.feedback,
        human_feedback=None
    )
    
    return {
        "is_verified": result.correct,
        "verification_feedback": result.feedback,
        "needs_hitl": result.needs_hitl
    }