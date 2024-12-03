import requests as r
import os
from langchain_openai import AzureOpenAIEmbeddings
from openai import AzureOpenAI
from azure.search.documents.models import VectorizedQuery
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential

print(os.getenv("AZURE_OPENAI_API_VERSION"))

embeddings_client = AzureOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY")
)

search_client = SearchClient(
    endpoint=os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT"), 
    index_name=os.getenv("AZURE_SEARCH_INDEX_NAME"), 
    credential=AzureKeyCredential(os.getenv("AZURE_SEARCH_API_KEY"))
)

query = 'hola soy Pat'


embedding = embeddings_client.embeddings.create(input=query, model=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"), dimensions=1536).data[0].embedding

print(embedding)

vector_query = VectorizedQuery(vector=embedding, k_nearest_neighbors=3, fields="contentVector")
  
results = search_client.search(  
    search_text=None,  
    vector_queries= [vector_query],
    select=["file_name", "content", "page", "link"],
)  
  
for result in results:  
    print(f"Title: {result['file_name']}")  
    print(f"Score: {result['@search.score']}")  
    print(f"Content: {result['content']}")  
    print(f"Page: {result['page']}\n")  