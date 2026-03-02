from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
from .llm import llm
from .state import AgentState

class IntentRouterAgentOutput(BaseModel):
    intent: str = Field(..., description="The identified intent of the problem" \
                    "(eg. 'solve_math': The user wants to solve a specific math problem." \
                    "'explain_concept': The user is asking for a general conceptual explanation." \
                    "'out_of_scope': The query is unrelated to JEE Mathematics (e.g., casual chat, biology, history).)")
    
model = llm()

SYSTEM_PROMPT = f"""
You are a helpful assistant that identifies the intent of a user's query.
The intent can be one of three categories:
1. 'solve_math': The user wants to solve a specific math problem.
2. 'explain_concept': The user is asking for a general conceptual explanation.
3. 'out_of_scope': The query is unrelated to JEE Mathematics (e.g., casual chat, biology, history).
"""

agent = create_agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
    response_format=IntentRouterAgentOutput
)

def intent_router_agent_node(state: AgentState):
    text_input = state.get("text_input", "")
    text_input = text_input.replace("\n", " ").strip()
    topic = state.get("parsed_topic", "")
    variables = state.get("parsed_variables", [])
    constraints = state.get("parsed_constraints", [])

    user_input = f"""
    Here is the user's query: {text_input}
    The identified topic is: {topic}
    The identified variables are: {variables}
    The identified constraints are: {constraints}
    Based on this information, determine the user's intent. Is it 'solve_math', 'explain_concept', or 'out_of_scope'?
    """

    response = agent.invoke({
            "messages": [HumanMessage(content=user_input)]
        })
    result = response["structured_response"]
    
    return {
        **state,
        "intent": result.intent
    }