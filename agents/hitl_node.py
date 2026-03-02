from langgraph.types import interrupt
from .state import AgentState
from utils.memory_bank import save_memory

def hitl_node(state: AgentState):
    
    human_review = interrupt({
        "action": "Human Review Required",
        "current_state": {
            "input": state.get("text_input", ""),
            "solution": state.get("solution", ""),
            "feedback": state.get("verification_feedback", "")
        }
    })
    
    decision = human_review.get("type")
    
    if decision in ["approve", "edit"]:
        final_solution = human_review.get("edited_solution", state.get("solution", ""))
        
        save_memory(
            original_input=state.get("text_input", ""),
            retrieved_context=state.get("retrieved_context", ""),
            final_answer=final_solution,
            verifier_outcome=state.get("verification_feedback", "Approved"),
            human_feedback=decision
        )
            
        return {
            "solution": final_solution,
            "is_verified": True,
            "needs_hitl": False,
            "needs_clarification": False
        }
        
    elif decision == "reject":
        return {
            "solution": "Solution rejected by human reviewer. Please try again or rephrase the problem.",
            "is_verified": False,
            "needs_hitl": False
        }
    
    return {}