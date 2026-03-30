from langchain_core.messages import SystemMessage,AnyMessage
from langchain.messages import ToolMessage
from langgraph.graph import StateGraph, START, END

from typing import Literal
from tools import model_with_tools, tool_by_name
from typing_extensions import List, Annotated, TypedDict
import operator
import logging


logging.basicConfig(level = logging.DEBUG, format ='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

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
    logger.info("LLM call ..")    
    return {
        "messages":[
            model_with_tools.invoke(
                [
                    SystemMessage(
                        content = """You are a helpful assistant that performs arithmetic using external tools.
                                Think step by step analyze the question. If it have many questions, thinking and break down to small steps if need.
                                Each steps always decide whether to call a tool or not based on the question and your own knowledge.
                                If use tool choose the most suitable documents that extracted from tool to answer the question.
                                If use the tool get the results, you should use remain results to answer the question. Don't add more information. Else use your own knowledge"""
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
    logger.info("Node: tool_node....")
    result = []
    final_response = state.get("final_response","")

    for tool_call in state['messages'][-1].tool_calls:
        tool = tool_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call['args'])
        if tool_call["name"] == "retrieval":
            final_response = observation
        result.append(ToolMessage(content = observation, tool_call_id = tool_call['id']))

    return {"messages":result, "final_response":final_response}