
from langgraph.graph import StateGraph, START, END
from langchain.messages import HumanMessage
from nodes import should_continue, tool_node, llm_call, State

#define workflow
agent_workflow = StateGraph(State)
agent_workflow.add_node("llm_call", llm_call)
agent_workflow.add_node("tool_node", tool_node)
agent_workflow.add_edge(START,"llm_call")
agent_workflow.add_conditional_edges(
    "llm_call", should_continue, ["tool_node", END]
)
agent_workflow.add_edge("tool_node", "llm_call")

agent = agent_workflow.compile()


#-------------------------

messages = [HumanMessage(content = "what is machine learning ? what is love ?")]
resp  = agent.invoke({"messages":messages})
for m in resp['messages']:
    m.pretty_print()
