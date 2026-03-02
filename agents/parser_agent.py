from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from .llm import llm
from pydantic import BaseModel, Field
from .state import AgentState

class ParserAgentOutput(BaseModel):
    problem_text: str = Field(..., description="The problem statement extracted from the raw input, cleaned and formatted.")
    topic: str = Field(..., description="The identified topic of the problem (e.g. 'algebra', 'calculus', 'probability', etc.)")
    variables: list[str] = Field(..., description="A list of variables mentioned in the problem. eg. ['x', 'y']")
    constraints: list[str] = Field(..., description="A list of any constraints or conditions mentioned in the problem. eg ['x > 0']")
    needs_clarification: bool = Field(..., description="A boolean indicating whether the problem statement is clear or if it requires further clarification. eg. True or False")

model = llm()

SYSTEM_PROMPT = f"""
You are a helpful assistant that takes raw text input from OCR or ASR and formats it into a structured format as given. 
"""

agent = create_agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
    response_format=ParserAgentOutput
)

def parser_agent_node(state: AgentState):
    input_text = state.get("text_input", "")
    input_text = input_text.replace("\n", " ").strip()
    user_input = f"Here is the raw text input: {input_text}. Format this into the specified structure."

    response = agent.invoke({
            "messages": [HumanMessage(content=user_input)]
        })
    result = response["structured_response"]
    
    return {
        **state,
        "parsed_topic": result.topic,
        "parsed_variables": result.variables,
        "parsed_constraints": result.constraints,
        "needs_clarification": result.needs_clarification
    }

