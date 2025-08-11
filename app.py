import base64
# Helper function to convert image to base64
def get_image_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()
import streamlit as st
import asyncio
import core
from dotenv import load_dotenv
import time
from PIL import Image


logo = Image.open("logo.png")
# Use logo.png for branding

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="CysticCare AI - AI Support for Polycystic Kidney Disease",
    page_icon=logo,
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2E86AB;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #2E86AB;
        background-color: #f8f9fa;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left-color: #1976d2;
    }
    .assistant-message {
        background-color: #f1f8e9;
        border-left-color: #388e3c;
    }
    .stChatMessage {
        background-color: transparent;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_initialized" not in st.session_state:
    st.session_state.session_initialized = False

async def initialize_chat_session():
    """Initialize the CysticCare AI session"""
    if not st.session_state.session_initialized:
        try:
            await core.initialize_session()
            st.session_state.session_initialized = True
            return True
        except Exception as e:
            st.error(f"Failed to initialize session: {e}")
            return False
    return True

async def get_chatpkd_response(user_input):
    """Get response from CysticCare AI agent"""
    try:
        response = await core.agent_response(user_input)
        return response
    except Exception as e:
        st.error(f"Error getting response: {e}")
        return "Sorry, I encountered an error while processing your request. Please try again."

def main():
    # Branding: Show logo next to the main header
    # Centered logo and header using flexbox
    st.markdown(
        f'''
        <div style="display: flex; align-items: center; justify-content: center; gap: 16px; margin-bottom: 1rem;">
            <img src="data:image/png;base64,{get_image_base64('logo.png')}" alt="Logo" style="width:60px; height:auto;" />
            <h1 class="main-header" style="margin: 0;">CysticCare AI</h1>
        </div>
        ''', unsafe_allow_html=True
    )
    st.markdown('<p class="subtitle">AI Support Agent for Polycystic Kidney Disease</p>', unsafe_allow_html=True)
    
    # Description
    with st.expander("ℹ️ About CysticCare AI", expanded=False):
        st.markdown("""
        **CysticCare AI** is an AI-powered support agent designed to help patients with Polycystic Kidney Disease (PKD).
        
        **What you can ask:**
        - Questions about PKD symptoms and management
        - Treatment options and lifestyle recommendations
        - Support and guidance for living with PKD
        - General information about kidney health
        
        **Important:** This AI assistant provides general information and support. Always consult with your healthcare provider for medical advice and treatment decisions.
        """)
    
    # Initialize session
    if not st.session_state.session_initialized:
        with st.spinner("Initializing CysticCare AI session..."):
            success = asyncio.run(initialize_chat_session())
            if success:
                st.success("✅ CysticCare AI is ready to help!")
                time.sleep(1)
                st.rerun()
    
    # Chat interface
    st.markdown("### 💬 Chat with CysticCare AI")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask CysticCare AI about Polycystic Kidney Disease..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("CysticCare AI is thinking..."):
                response = asyncio.run(get_chatpkd_response(prompt))
            st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Sidebar with additional features
    with st.sidebar:
        st.markdown("### 🛠️ Options")
        
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
        
        if st.button("🔄 Reset Session"):
            st.session_state.session_initialized = False
            st.session_state.messages = []
            st.rerun()
        
        st.markdown("---")
        st.markdown("### 📋 Quick Questions")
        quick_questions = [
            "What is Polycystic Kidney Disease?",
            "What are the symptoms of PKD?",
            "How is PKD diagnosed?",
            "What treatment options are available?",
            "How can I manage PKD symptoms?",
            "What lifestyle changes can help with PKD?"
        ]
        
        for question in quick_questions:
            if st.button(question, key=f"quick_{question}"):
                # Add the quick question as if user typed it
                st.session_state.messages.append({"role": "user", "content": question})
                
                # Get response
                with st.spinner("ChatPKD is thinking..."):
                    response = asyncio.run(get_chatpkd_response(question))
                
                # Add response to chat
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()
        
        st.markdown("---")
        st.markdown("### ⚠️ Disclaimer")
        st.markdown("""
        <small>
        This AI assistant provides general information only. 
        Always consult healthcare professionals for medical advice.
        </small>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
