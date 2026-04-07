from langchain.tools import tool
from langchain.chat_models import init_chat_model
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from ragas.llms import llm_factory
from ragas.metrics import DiscreteMetric
from openai import OpenAI
from dotenv import load_dotenv
from qdrant_client import QdrantClient
import os 
import logging

load_dotenv(override=True)


logging.basicConfig(level = logging.DEBUG, format ='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

class ToolNodes:
  def __init__(self, embedding_model):
    self.embedding_model = embedding_model
    self.qdrant_client = QdrantClient(
        url = os.getenv("QDRANT_URL"),
        api_key = os.getenv("QDRANT_API_KEY"),
        cloud_inference = True
    )
  
  #define tool function

  def retrieval(self,query: str):
    """Documents about machine learning , RAG and love. 
        if user ask some question related to machine learning, RAG or love, please use the tool first.
        Extract relevant documents
      args:
        query: user's query 
    """
    logger.info("Start extract documents")
    documents = [
      "Machine learning is a subset of artificial intelligence (AI) that enables computers to learn from data and improve their performance over time without being explicitly programmed for every specific task",
      "Love is messy",
      "Retrieval-Augmented Generation (RAG) is an AI framework that improves Large Language Model (LLM) accuracy by retrieving data from external, trusted knowledge bases (documents, databases, internet) before generating a response"
    ]
    embeddings_documents = self.embedding_model.encode(documents)
    embeddings_query = self.embedding_model.encode(query)
    dim = 1024
    embed_query = np.array(embeddings_documents.reshape(-1,dim))
    embed_documents = np.array(embeddings_query.reshape(-1,dim))
    cosine_similarity_all_documents = np.array(cosine_similarity(embed_query, embed_documents))
    index = np.argsort(cosine_similarity_all_documents.flatten().tolist()) #find the most relevant document
    doc = [documents[i] for i in index]
    doc = doc[::-1]
    return doc[0]

  def extract_legal_document(self, query: str):
    """Extract relevant legal documents based on the user's query.
      args:
        query: users' question
      """
    logger.info("Start extract legal documents")
    documents = []
    encoded_query = self.embedding_model.encode(query)
    hits = self.qdrant_client.query_points(
        collection_name = "legal",
        query = encoded_query,
        limit = 3
    )
    for hit in hits.points:
        documents.append(hit.payload['text'])
    
    return documents[0]

    
  def eval_response(self,query:str, documents:str):
    """After extract relevant documents, evaluates each document to determine its relevance to the user's query before proceeding to the next step.
      Args:
        query: user's query
        documents: relevant documents
      
      """
    logger.info("Eval extracted documents with user's question")
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
  

