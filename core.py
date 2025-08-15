import os
from pygame import mixer
import pygame

from dotenv import load_dotenv
import time
import threading
import asyncio

from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.events import Event
from google.genai import types

import faiss
import numpy as np

from sentence_transformers import SentenceTransformer
# sentence_transformer class is apparently built on top of HugginsFace Transformer

import pdfplumber

import openai

"""
Future developers note:
 - This is the core script for ChatPKD
 - To edit the instructions fed into the agent, edit the instructions.txt file in this folder.
    there is no need to edit this file for that purpose.


 Future TODO:
  - Instead of feeding raw text instructions, by asking chatgpt to create summaries
    from trusted scientific papers to and then feeding that to this agent,
    we can vectorize the scientific papers to feed that into chatgpt.
    this would make the entire process more efficient and accurate.

 - instead of returning a text, we could use text to speech api to read the text out loud.


"""


load_dotenv()

api_key = os.getenv("OPEN_AI_API_KEY")


folder = 'papers'
chunks =[]
import re

#Code to chunk up the code(break it into multiple piecies so that Litellm doesnt overload with text)
def chunk_up_context(text, chunk_length = 400):
    words = re.findall(r'\S+', text) 
    chunks = []
    for i in range(0, len(words), chunk_length):
        chunk = ' '.join(words[i:i + chunk_length])
        chunks.append(chunk)
    
    return chunks

# the below for loop is for the python script to read the pdf's in the papers folder and break it into chunks
for filename in os.listdir(folder):
    if filename.endswith('.pdf'):
        with pdfplumber.open(os.path.join(folder,filename)) as pdf:
            text =''

            for page in pdf.pages:
                text+=page.extract_text() or ""
            chunked_context = chunk_up_context(text)
            for i, para in enumerate(chunked_context):
                chunks.append({"text":para, "source": filename, "chunk_index": i })



# below 3 lines of code basically converts all the text pieces from the papers folder and converts them into vector
model = SentenceTransformer('all-MiniLM-L6-v2')
texts= [chunk["text"] for chunk in chunks]
embeddings = model.encode(texts)


#The below lines of code basically stores all the embeddings created in the last couple of lines and stores them in a Vector Database
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(np.array(embeddings))


if not api_key:
    print("Error: OPEN_AI_API_KEY not found in .env file.")
    raise ValueError("OPEN_AI_API_KEY not found in .env file.")

# Set the environment variable that LiteLLM expects
os.environ["OPENAI_API_KEY"] = api_key

client = openai.OpenAI(api_key=api_key)

authentication_event = threading.Event()


main_agent = LlmAgent(
    name='ChatPKD',
    model=LiteLlm(model="gpt-4o-mini"),
    description="ChatPKD is a AI support agent that can aid patients with Polycystic Kidney Disease, a genetic disorder characterized by the growth of numerous fluid-filled cysts in the kidneys, potentially leading to kidney failure" 
)

service = InMemorySessionService()

# Initialize session variables
SESSION_ID = None
session = None

async def initialize_session():
    global SESSION_ID, session
    SESSION_ID = f"session_{int(time.time())}"
    session = await service.create_session(
        app_name = "ChatPKD Session",
        user_id = "user_main",
        session_id=SESSION_ID
    )
    print(f"Created session: {SESSION_ID}")
    return session

runner = Runner(
    agent=main_agent,
    app_name = "ChatPKD Session",
    session_service=service,    
)

async def agent_response(input: str):
    global SESSION_ID

    input = input + "Use plain, everyday language that anyone can understand.Avoid technical terms or specialized vocabulary. Keep responses concise under 150 words unless the user wants you to explain the question in detail. Aim for clarity, simplicity, and easy readability for a general audience. If you cannot answer from the context, reply: “Sorry unable to provide the answer. The question that you asked is outside my knowledge base. I am a chatbot designed only to answer questions about Polycystic Kidney Disease”"
    
    query_emb = model.encode([input])

    # D is the distance between the closest vectors
    # I is the indecies of the 1 closest vector
    D,I = index.search(np.array(query_emb) , k =1)

    context = "\n\n".join([chunks[i]['text'] for i in I[0]])

    full_input__for_LLM = f"Context:/n{context}\n\n Question: {input}"


    answer = "Sorry, I had trouble understanding you. Please try again."
    agent_name = runner.agent.name
    final_agent_output = None
    
    # Check if session is initialized
    if not SESSION_ID:
        print("Session not initialized. Initializing now...")
        await initialize_session()
    
    try:
        #the below line of code is used to format the input text into a content object
        content = types.Content(role = "user", parts=[types.Part(text=full_input__for_LLM)])

        async for event in runner.run_async(
            user_id="user_main", session_id=SESSION_ID, new_message=content
        ):
            if event.content and event.content.parts:
                for i, part in enumerate(event.content.parts):
                    if event.author == agent_name:
                        final_agent_output = event.content.parts[-1].text 
            
            if event.actions:
                if hasattr(event.actions, 'tool_code_outputs'):
                    print(f"    Tool Code Outputs: {event.actions.tool_code_outputs}")
                    
                if hasattr(event.actions, 'tool_code_invocation'):
                    print(f"    Tool Code Invocation: {event.actions.tool_code_invocation}")


        if final_agent_output:
            answer = final_agent_output
            print(f"ChatPKD: {answer}")
        
    except Exception as e:
        print(f"Error in agent_response: {e}")
        print(f"Session ID: {SESSION_ID}")
        print("Sorry. something went wrong.")    
    
    return answer

def close_application():
    #this function is for future where I will use pygame api to convert text back to speech
    """
    if pygame.mixer.get_init():
        pygame.mixer.quit()
    """

    print("Closed app. Visit again!")