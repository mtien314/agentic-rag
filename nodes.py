from langchain_core.messages import SystemMessage,AnyMessage
from langchain.messages import ToolMessage
from langgraph.graph import StateGraph, START, END

from typing import Literal
from tools import model_with_tools, tool_by_name
from typing_extensions import List, Annotated, TypedDict
import operator


class State(TypedDict):
    messages: Annotated[List[AnyMessage],operator.add]
    llm_call: int
    final_response: str


#define func for edge

def llm_call(state: dict):
    """LLM decide whether to call a tool or not"""

    #last_messages = state['messages'][-1]
    # if last_messages.content == "pass":
    #     #finish dont need you llm
    #     return 
    
    return {
        "messages":[
            model_with_tools.invoke(
                [
                    SystemMessage(
                        content = """You are a helpful assistant that performs arithmetic using external tools.
                                Rules:
                                1. If tool results are available, use them as the final answer.
                                2. Do NOT override or recompute tool outputs.
                                3. If the tool does NOT provide a result or returns empty/invalid output,
                                then use your own knowledge to generate the answer.

                                Always prioritize tool outputs over your own reasoning.

                                """
                    )
                ]
                + state["messages"]
            )
        ],
        "llm_calls": state.get("llm_calls", 0)+1
    }



#conditional edge function to route to the tool node or end base upon whether the LLM made

def should_continue(state: State) -> Literal["tool_node", END]: #only return tool_node or end
    """decide if we should continue the loop or stop based upon wheather the LLM"""
    messages = state['messages']
    last_message = messages[-1]    

    #if the LLM makes a tool call then perform an action

    if last_message.tool_calls:
        return "tool_node"
    
    return END



def tool_node(state: dict):
    """Performs the tool calls"""
    result = []
    final_response = state.get("final_response","")

    for tool_call in state['messages'][-1].tool_calls:
        print(tool_call)
        tool = tool_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call['args'])
        if tool_call["name"] == "retrieval":
            final_response = observation
        result.append(ToolMessage(content = observation, tool_call_id = tool_call['id']))
    return {"messages":result, "final_response":final_response}