from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.core.credentials import AzureKeyCredential
from flask import Flask, render_template, request, jsonify
import openai
import os
from dotenv import load_dotenv

load_dotenv()
# Initialize LLM model
openai_client = openai.AzureOpenAI(
  azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"), 
  api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
  api_version=os.getenv("AZURE_OPENAI_API_VERSION")
)

# Initialize embeddings model
embeddings_client = openai.AzureOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY")
)

# Initialize search index
search_client = SearchClient(
    endpoint=os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT"), 
    index_name=os.getenv("AZURE_SEARCH_INDEX_NAME"), 
    credential=AzureKeyCredential(os.getenv("AZURE_SEARCH_API_KEY"))
)

# Perform vector search to retrieve top 5 relevant chunks
def vector_search(query, embeddings_client, search_client):

    embedded_query = embeddings_client.embeddings.create(input=query, model=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")).data[0].embedding

    print(len(embedded_query))

    vector_query = VectorizedQuery(vector=embedded_query, k_nearest_neighbors=5, fields="contentVector")

    results = search_client.search(  
        search_text=None,  
        vector_queries= [vector_query],
        select=["file_name", "content", "page", "link"],
    )

    content_list = []  
    
    for result in results: 
        content_list.append(result) 
        print(f"Title: {result['file_name']}")  
        print(f"Score: {result['@search.score']}")  
        print(f"Content: {result['content']}")  
        print(f"Page: {result['page']}\n")  

    return content_list

# Call the OpenAI API with the top 5 chunks as context
def query_llm_with_context(context_chunks, user_query, openai_client):
    top_chunks = []
    for result in context_chunks:
            top_chunks.append(result['content'])
   
    # Join the chunks into a single context for the LLM
    context = "\n".join(top_chunks)
    user_prompt = f"Context:\n{context}\n\nQuestion: {user_query}\nAnswer:"

    system_prompt = build_system_prompt()

    messages = [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': user_prompt}]

    # Query the OpenAI GPT model
    response = openai_client.chat.completions.create(
                model=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
                messages=messages,
                temperature=0.5
            )


    answer = {'name' : 'Asistente',
            'content' : response.choices[0].message.content,
            'context' : context_chunks
            }
    
    return answer

# Define system prompt
def build_system_prompt() -> str:
    """
    Builds the role prompt passed to the system.
    """
    role_prompt = "Eres un agente capaz de responder a preguntas de forma ordenada, clara y de forma certera. \
        Te voy a proporcionar un una serie de bloques de informacion objetivos como contexto y en base a ellos, tu objetivo es responder a la pregunta en cuestion. \
        Utiliza un tono formal y devuleve tu respuesta en formato HTML devolviendo unicamnete el contenido del body; ordenando y estructurando tu respuesta para que sea facil de leer y siempre objetiva y basada en el contexto proporcionado. \
        En caso de no tener informacion para responder a la pregunta, ya sea por falta de contenido o por ser una pregunta poco relacionada con el contexto, debes indicar que no tienes la suficiente informacion para responder a la pregunta. \
        No intentes inventarte respuestas."

    return role_prompt

# Start application 
app = Flask(__name__)
app.config['DEBUG'] = True

@app.route('/')
def chatbot_interface():
    return render_template('chatbot.html')

@app.route('/ask', methods=['POST'])
def get_response():
    user_query = request.json['message']
    if user_query:
        top_chunks = vector_search(query=user_query, embeddings_client=embeddings_client, search_client=search_client)

        answer = query_llm_with_context(top_chunks, user_query, openai_client)

        return jsonify(answer)
    else: 
        return jsonify({"response": "No question provided"})

    


# # Azure Search Configuration
# service_name = "your-azure-service-name"
# index_name = "your-index-name"
# api_key = "your-azure-api-key"

# # OpenAI API Key
# openai_api_key = "your-openai-api-key"

# # Initialize the Azure Search Client
# client = get_azure_search_client(service_name, index_name, api_key)

# Call the chatbot interface
if __name__ == "__main__":
    app.run(debug=True)
