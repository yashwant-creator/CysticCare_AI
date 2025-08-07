# ChatPKD - Streamlit Web Interface

A web-based AI support agent for Polycystic Kidney Disease (PKD) patients, built with Streamlit and powered by GPT-4o-mini.

## Features

- 🩺 AI-powered chat interface for PKD support
- 💬 Real-time conversation with ChatPKD agent
- 📋 Quick question buttons for common queries
- 🔄 Session management and chat history
- 📱 Responsive web interface

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment variables:**
   Create a `.env` file with your OpenAI API key:
   ```
   OPEN_AI_API_KEY=your_openai_api_key_here
   ```

3. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

4. **Access the app:**
   Open your browser and go to `http://localhost:8501`

## Usage

- Type your questions about PKD in the chat input
- Use the quick question buttons in the sidebar for common queries
- Clear chat history or reset session using sidebar buttons
- All conversations are maintained during your session

## Files

- `app.py` - Main Streamlit application
- `core.py` - Core ChatPKD agent logic
- `instructions.txt` - AI agent instructions
- `main.py` - Original CLI interface (still available)

## Important Note

This AI assistant provides general information and support. Always consult with healthcare professionals for medical advice and treatment decisions.
