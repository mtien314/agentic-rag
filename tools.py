from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langchain_groq import ChatGroq
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq
from ragas.llms import llm_factory
from ragas.metrics import DiscreteMetric
from openai import OpenAI
from dotenv import load_dotenv
from huggingface_hub import login
from sentence_transformers import SentenceTransformer
import os 
load_dotenv()

access_token = os.getenv("ACCESS_TOKEN")
model_name = "openai/gpt-oss-20b"
model = init_chat_model(
    model_name,
    model_provider="Groq",
    temperature  = 0
)

#load embeddings model
login(access_token)
embedding_model = SentenceTransformer("google/embeddinggemma-300m")


#define tools

# @tool
# def mult(a: int, b: int) -> int:
#   """" multiply `a` and `b`
#   args:
#     a: first int
#     b: second int
#   """
#   return a*b

# @tool
# def add(a: int, b: int) -> int:
#   """ Adds `a` and `b`
#   args:
#     a: first int
#     b: first int
#   """
#   return a+b


# @tool
# def area_square(a: int) ->int:
#   """Calculate area square with `a` 
#     Args:
#      a: edge
#   """
#   return a*a

# @tool
# def area_rectangle(a: int, b:int ) -> int:
#   """Calculate area rectangle
#    args:
#       a:length
#       b: width"""
#   return a*b

@tool
def retrieval(query: str):
  """Document about machine learning , RAG.
    retrieval relevant documents
    args:
      query: user's query
  """

  documents = [
    "Machine learning is a subset of artificial intelligence (AI) that enables computers to learn from data and improve their performance over time without being explicitly programmed for every specific task",
   # "Love is messy",
    "Retrieval-Augmented Generation (RAG) is an AI framework that improves Large Language Model (LLM) accuracy by retrieving data from external, trusted knowledge bases (documents, databases, internet) before generating a response"
  ]
  embeddings_documents = embedding_model.encode(documents)
  embeddings_query = embedding_model.encode(query)
  dim = 768
  embed_query = np.array(embeddings_documents.reshape(-1,dim))
  embed_documents = np.array(embeddings_query.reshape(-1,dim))
  cosine_similarity_all_documents = cosine_similarity(embed_query, embed_documents)
  index = np.argmax(cosine_similarity_all_documents) #find the most relevant document
  doc = documents[index]
  return doc

@tool
def summary(query: str, doc: str):
  """After eval relevant document with user's query, summary and return final response base on user's query and relevant document
  args:
    query: user's query
    doc: relevant documents
  """
  prompt = f"Base on user's query {query} and relevant documents {doc} summary and response final answer"
  response = model.invoke(prompt)
  return response.content

@tool
def eval_response(query:str, documents:str):
  """After extract relevant documents, evaluates each document to determine its relevance to the user's query before proceeding to the next step.
    Args:
      query: user's query
      documents: relevant documents
    
    """
  api_key = os.getenv("GROQ_API_KEY")
  client = OpenAI(
      api_key=api_key,
      base_url="https://api.groq.com/openai/v1",
  )

  llm = llm_factory("openai/gpt-oss-120b", client=client)
  my_metric = DiscreteMetric(
      name="correctness",
      prompt = "Check if the response relevant from the grading notes and return 'pass' or 'fail'.\nResponse: {response} Grading Notes: {grading_notes}",
      allowed_values=["pass", "fail"],
  )
  score = my_metric.score(
        llm=llm,
        response=documents,
        grading_notes=query
    )
  return score.value
  

#augment the llm with tools
tools = [retrieval]
tool_by_name = {tool.name:tool for tool in tools}
model_with_tools = model.bind_tools(tools)
