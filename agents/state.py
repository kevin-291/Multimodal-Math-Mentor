from typing import TypedDict, Optional, List

class AgentState(TypedDict):
    """Shared state across all agents"""
    id: str
    text_input: str
    confidence: Optional[float]

    #Parser agent outputs
    parsed_topic: Optional[str] 
    parsed_variables: Optional[List[str]]
    parsed_constraints: Optional[List[str]]
    needs_clarification: Optional[bool]

    #Intent router agent output
    intent: Optional[str]

    #Solver agent output
    retrieved_context: Optional[str]
    past_similar_problems: Optional[str]
    solution: Optional[str]

    #Critic agent output
    is_verified: Optional[bool]
    verification_feedback: Optional[str]
    needs_hitl: Optional[bool]

    #Tutor agent output
    tutor_explanation: Optional[str]

    #HITL
    user_requests_recheck: Optional[bool]