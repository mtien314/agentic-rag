from langgraph.graph import StateGraph, START, END
from langchain.messages import HumanMessage
from nodes import Edge, State
from sentence_transformers import SentenceTransformer


#load embedding model
embedding_model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B") #choose model size lightweight


edge = Edge(embedding_model=embedding_model)
#define workflow
agent_workflow = StateGraph(State)
agent_workflow.add_node("llm_call", edge.llm_call)
agent_workflow.add_node("tool_node", edge.tool_node)
agent_workflow.add_edge(START,"llm_call")
agent_workflow.add_conditional_edges(
    "llm_call", edge.should_continue, ["tool_node", END]
)
agent_workflow.add_edge("tool_node", "llm_call")

agent = agent_workflow.compile()


#-------------------------

messages = [HumanMessage(content = "nhiệm vụ của lực lượng tuần tra")]
resp  = agent.invoke({"messages":messages})
for m in resp['messages']:
    m.pretty_print()
