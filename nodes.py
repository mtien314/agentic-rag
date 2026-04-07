from langchain_core.messages import SystemMessage,AnyMessage
from langchain_core.tools import StructuredTool
from langchain.messages import ToolMessage
from langgraph.graph import StateGraph, START, END
from langchain.chat_models import init_chat_model
from typing import Literal
from tools import ToolNodes
from typing_extensions import List, Annotated, TypedDict
import operator
import logging
import backoff

logging.basicConfig(level = logging.DEBUG, format ='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

class State(TypedDict):
    messages: Annotated[List[AnyMessage],operator.add]
    llm_call: int


class Edge:
    def __init__(self, embedding_model):
        self.tool_class = ToolNodes(embedding_model=embedding_model)
        model =  init_chat_model(
            model = "openai/gpt-oss-120b",
            model_provider="Groq",
            temperature  = 0
            )
        #self.tools = [tool_class.extract_legal_document, tool_class.eval_response]
        self.tools = [
            StructuredTool.from_function(
                lambda query: self.tool_class.extract_legal_document(query),
                name="extract_legal_document",
                description = "Extract relevant legal documents based on the user's query."
            ),
            StructuredTool.from_function(
                lambda query, documents: self.tool_class.eval_response(query, documents) ,
                name="eval_response",
                description = "After extract relevant documents, evaluates each document to determine its relevance to the user's query."
            )
        ]

        self.model_with_tools = model.bind_tools(self.tools)
        self.tool_by_name = {tool.name: tool for tool in self.tools}
    
    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    def llm_call(self, state: dict):
        """LLM decide whether to call a tool or not"""

        logger.info("LLM call ..")    

        return {
            "messages":[
                self.model_with_tools.invoke(
                    [
                        SystemMessage(
                            content = """ You are a reliable AI assistant that solves user queries using step-by-step reasoning and external tools when needed.

                                You MUST follow this pipeline strictly:

                                1. UNDERSTAND
                                - Parse the question.
                                - If multiple sub-questions exist, break them into smaller steps.

                                2. PLAN
                                - Decide for each step:
                                  - Can it be answered from your own knowledge?
                                  - Or requires a tool?

                                3. TOOL USAGE (if needed)
                                - Call the most relevant tool.
                                - Only extract necessary information from tool results.
                                - DO NOT hallucinate or add extra info beyond tool results.

                                4. EVALUATION
                                - Check if the extracted documents is complete and correctly addresses all parts of the question.
                                - If tool result is insufficient:
                                  - Rewrite and retry retrieval with a better query OR
                                  - Fallback to your own knowledge (state clearly).
                                - Else: don't need to retry retrieval and rewrite

                                5. FINAL ANSWER
                                  - If tool results are available:
                                  - You MUST ONLY use tool results to answer.
                                  - DO NOT add, infer, or supplement with your own knowledge.
                                - If tool results are NOT available:
                                  - You may use your own knowledge.

                                Rules:
                                - Prefer tool results over your own knowledge when available.
                                - Never fabricate tool outputs.
                                - Be deterministic and structured."""
                        )
                    ]
                    + state["messages"]
                )
            ],
            "llm_calls": state.get("llm_calls", 0)+1
        }



    #conditional edge function to route to the tool node or end base upon whether the LLM made

    def should_continue(self, state: State) -> Literal["tool_node", END]: #only return tool_node or end
        """decide if we should continue the loop or stop based upon wheather the LLM"""
        messages = state['messages']
        last_message = messages[-1]    

        #if the LLM makes a tool call then perform an action

        if last_message.tool_calls:
            return "tool_node"
        
        return END

    def tool_node(self, state: dict):
        """Performs the tool calls"""
        logger.info("Node: tool_node....")
        result = []
        for tool_call in state['messages'][-1].tool_calls:
            tool = self.tool_by_name[tool_call["name"]]
            logger.info(f"Tool:{tool}")
            observation = tool.invoke(tool_call['args'])
           
            result.append(ToolMessage(content = observation, tool_call_id = tool_call['id']))

        return {"messages":result}