from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from .llm import llm
from .state import AgentState

model = llm()

SYSTEM_PROMPT = f"""
You are an encouraging, expert JEE Mathematics Tutor.
You will receive the original problem, the mathematical solution, and notes from a Critic (Verifier).

Your task is to create a step-by-step, student-friendly explanation.
- Adopt a supportive, pedagogical tone. Explain *why* each step is taken, not just *how*.
- Highlight key JEE concepts or common pitfalls associated with this problem type.
- Format all mathematical equations using LaTeX (use $ for inline equations and $$ for display equations, with no spaces between the $ and the formula).
- eg. To solve for x, we can use the quadratic formula: $$x = \\frac{{-b \\pm \\sqrt{{b^2 - 4ac}}}}{{2a}}$$
- Do not encode the Latex in a list or code block. Just provide the raw LaTeX in the text.
"""

agent = create_agent(
    model=model,
    system_prompt=SYSTEM_PROMPT
)

def tutor_agent_node(state: AgentState):
    problem = state.get("text_input", "")
    solution = state.get("solution", "")
    critic_notes = state.get("verification_feedback", "")

    user_input = f"""
    Here is the original problem: {problem}
    
    Here is the mathematical solution: {solution}
    
    Here are the critic's notes: {critic_notes}
    
    Create a step-by-step, student-friendly explanation of the solution. Use a supportive tone and format equations in LaTeX.
    """

    response = agent.invoke({
            "messages": [HumanMessage(content=user_input)]
        })
    
    explanation = response["messages"][-1].content

    return {
        **state,
        "tutor_explanation": explanation
    }