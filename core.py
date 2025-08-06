import os
from pygame import mixer
import pygame
import openai
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
    instruction=open("instructions.txt").read(),
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
    
    answer = "Sorry, I had trouble understanding you. Please try again."
    agent_name = runner.agent.name
    final_agent_output = None
    
    # Check if session is initialized
    if not SESSION_ID:
        print("Session not initialized. Initializing now...")
        await initialize_session()
    
    try:
        #the below line of code is used to format the input text into a content object
        content = types.Content(role = "user", parts=[types.Part(text=input)])

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