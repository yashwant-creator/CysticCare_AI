from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from contextlib import asynccontextmanager
from typing import List
import uuid

# Your existing imports - exactly as in your original code
import os
from pathlib import Path
# pygame imports removed for FastAPI
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
import pdfplumber
import openai
import PyPDF2
import re

# Your existing global variables
BASE_DIR = Path(__file__).resolve().parent

# Load .env from the same directory as this file to avoid CWD issues
load_dotenv(dotenv_path=str(BASE_DIR / '.env'))
api_key = os.getenv("OPEN_AI_API_KEY")

# Resolve papers folder relative to this file regardless of where the app is launched
folder = str(BASE_DIR / 'papers')
chunks = []
model = None
index = None
runner = None
service = None
client = None
authentication_event = None
main_agent = None
SESSION_ID = None
session = None

# Pydantic models for API
class ChatRequest(BaseModel):
    message: str
    session_id: str

class ChatResponse(BaseModel):
    response: str
    source_titles: List[str]
    source_authors: List[str]

class SessionResponse(BaseModel):
    session_id: str
    message: str

# my exact existing functions
def chunk_up_context(text, chunk_length=400):
    words = re.findall(r'\S+', text) 
    chunks = []
    for i in range(0, len(words), chunk_length):
        chunk = ' '.join(words[i:i + chunk_length])
        chunks.append(chunk)
    return chunks

# the below fucntion is to extract the meta data to show it to the user as a source that the 
# chatbot used to explain the question
def extract_metadata(pdf_path):
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            metadata = reader.metadata
            if metadata is None:
                return 'Unknown Title', 'Unknown Author'
            title = metadata.get('/Title', 'Unknown Title')
            authors = metadata.get('/Author', 'Unknown Author')
            return title, authors
    except Exception as e:
        print(f"Error extracting metadata from {pdf_path}: {e}")
        return 'Unknown Title', 'Unknown Author'

async def initialize_agent_session():
    global SESSION_ID, session
    SESSION_ID = f"session_{int(time.time())}"
    session = await service.create_session(
        app_name="ChatPKD Session",
        user_id="user_main",
        session_id=SESSION_ID
    )
    print(f"Created session: {SESSION_ID}")
    return session

async def initialize_rag_system():
    """Initialize your existing RAG system exactly as in your original code"""
    global chunks, model, index, runner, service, client, authentication_event, main_agent, api_key
    
    print("🚀 Initializing RAG System...")
    
    # Your exact initialization code
    if not api_key:
        print("Error: OPEN_AI_API_KEY not found in .env file.")
        raise ValueError("OPEN_AI_API_KEY not found in .env file.")

    # Set the environment variable that LiteLLM expects
    os.environ["OPENAI_API_KEY"] = api_key
    client = openai.OpenAI(api_key=api_key)
    authentication_event = threading.Event()

    # Your exact PDF processing loop
    for filename in os.listdir(folder):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(folder, filename)
            title, authors = extract_metadata(pdf_path)
            with pdfplumber.open(pdf_path) as pdf:
                text = ''
                for page in pdf.pages:
                    text += page.extract_text() or ""
                chunked_context = chunk_up_context(text)
                for i, para in enumerate(chunked_context):
                    chunks.append({"text": para, "source": filename, "chunk_index": i, "title": title, "authors": authors})

    print(f"📚 Loaded {len(chunks)} chunks from PDFs")

    # Your exact embeddings creation
    model = SentenceTransformer('all-MiniLM-L6-v2')
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts)

    # Your exact FAISS index creation
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))

    # Your exact agent initialization
    main_agent = LlmAgent(
        name='ChatPKD',
        model=LiteLlm(model="gpt-4o-mini"),
        description="ChatPKD is a AI support agent that can aid patients with Polycystic Kidney Disease, a genetic disorder characterized by the growth of numerous fluid-filled cysts in the kidneys, potentially leading to kidney failure" 
    )

    service = InMemorySessionService()

    runner = Runner(
        agent=main_agent,
        app_name="ChatPKD Session",
        session_service=service,    
    )
    
    print("✅ RAG System initialized successfully")

async def agent_response(input: str):
    """Your exact existing agent_response function"""
    global SESSION_ID

    input = input + " Use plain, everyday language that anyone can understand. Avoid technical terms or specialized vocabulary. Keep responses concise under 250 words unless the user wants you to explain the question in detail. Aim for clarity, simplicity, and easy readability for a general audience. If you cannot answer from the context, reply: 'Sorry unable to provide the answer. The question that you asked is outside my knowledge base. I am a chatbot designed only to answer questions about Polycystic Kidney Disease.'"
    
    query_emb = model.encode([input])

    # D is the distance between the closest vectors
    # I is the indecies of the 1 closest vector
    D, I = index.search(np.array(query_emb), k=3)

    context = "\n\n".join([chunks[i]['text'] for i in I[0]])
    source_titles = [chunks[i]['title'] for i in I[0]]  # Retrieve titles
    source_authors = [chunks[i]['authors'] for i in I[0]]  # Retrieve authors

    full_input__for_LLM = f"Context:/n{context}\n\n Question: {input}"

    answer = "Sorry, I had trouble understanding you. Please try again."
    agent_name = runner.agent.name
    final_agent_output = None
    
    # Check if session is initialized
    if not SESSION_ID:
        print("Session not initialized. Initializing now...")
        await initialize_agent_session()
    
    try:
        # the below line of code is used to format the input text into a content object
        content = types.Content(role="user", parts=[types.Part(text=full_input__for_LLM)])

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
    
    # Include source title and author information in the response
    return answer, source_titles, source_authors

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Startup
    await initialize_rag_system()
    yield
    # Shutdown
    print("🛑 Shutting down...")

# Create FastAPI app
app = FastAPI(
    title="CysticCare AI Backend",
    description="AI-powered support agent for Polycystic Kidney Disease patients",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your Flutter app's domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Endpoints
@app.get("/")
async def root():
    return {"message": "CysticCare AI Backend is running!", "version": "1.0.0"}

@app.post("/initialize", response_model=SessionResponse)
async def initialize_session():
    """Initialize a new chat session"""
    session_id = str(uuid.uuid4())
    return SessionResponse(session_id=session_id, message="Session initialized")

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Send message and get AI response"""
    try:
        response, source_titles, source_authors = await agent_response(request.message)
        
        return ChatResponse(
            response=response,
            source_titles=source_titles,
            source_authors=source_authors
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "chunks_loaded": len(chunks)}

@app.get("/quick-questions")
async def get_quick_questions():
    """Get predefined quick questions for PKD"""
    questions = [
        "What is Polycystic Kidney Disease?",
        "What are the symptoms of PKD?",
        "How is PKD diagnosed?",
        "What treatment options are available?",
        "How can I manage PKD symptoms?",
        "What lifestyle changes can help with PKD?"
    ]
    return {"questions": questions}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)