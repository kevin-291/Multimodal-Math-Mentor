from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from agents.state import AgentState
from agents.parser_agent import parser_agent_node
from agents.intent_router_agent import intent_router_agent_node
from agents.solver_agent import solver_agent_node
from agents.critic_agent import verifier_agent_node
from agents.tutor_agent import tutor_agent_node
from agents.hitl_node import hitl_node

def route_after_parser(state: AgentState):
    confidence = state.get("confidence", 1.0)
    if confidence < 0.7 or state.get("needs_clarification", False):
        return "hitl_node"
    return "intent_router"

def route_after_router(state: AgentState):
    if state.get("intent") == "solve_math":
        return "solver"
    elif state.get("intent") == "explain_concept":
        return "tutor"
    else:
        return END

def route_after_critic(state: AgentState):
    if state.get("needs_hitl") == True or state.get("user_requests_recheck") == True:
        return "hitl_node"
    return "tutor"

def route_after_hitl(state: AgentState):
    if state.get("solution") and state.get("is_verified", False):
        return "tutor"
    return END

workflow = StateGraph(AgentState)

workflow.add_node("parser", parser_agent_node)
workflow.add_node("intent_router", intent_router_agent_node)
workflow.add_node("solver", solver_agent_node)
workflow.add_node("critic", verifier_agent_node)
workflow.add_node("hitl_node", hitl_node)
workflow.add_node("tutor", tutor_agent_node)

workflow.add_edge(START, "parser")
workflow.add_conditional_edges("parser", route_after_parser, {"hitl_node": "hitl_node", "intent_router": "intent_router"})
workflow.add_conditional_edges("intent_router", route_after_router, {"solver": "solver", "tutor": "tutor", END: END})
workflow.add_edge("solver", "critic")
workflow.add_conditional_edges("critic", route_after_critic, {"hitl_node": "hitl_node", "tutor": "tutor"})
workflow.add_conditional_edges("hitl_node", route_after_hitl, {"tutor": "tutor", END: END})
workflow.add_edge("tutor", END)

checkpointer = MemorySaver()
app = workflow.compile(
    checkpointer=checkpointer 
)