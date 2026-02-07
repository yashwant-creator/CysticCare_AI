# CysticCare AI

An intelligent chatbot powered by Retrieval-Augmented Generation (RAG) designed to assist people with Polycystic Kidney Disease (PKD). The system provides evidence-based answers by retrieving information from a curated collection of 80+ peer-reviewed research papers.

**🌐 Live Demo:** [https://cysticcare-ai.web.app/](https://cysticcare-ai.web.app/)

---

## 🧠 How It Works

### High-Level Architecture

```
User Question → Adaptive Agent → RAG Pipeline → OpenAI GPT-4 → Answer + Citations
                     ↓
            [CoT / Stepback Mode]
                     ↓
              Vector Database
           (80+ Research Papers)
```

### RAG (Retrieval-Augmented Generation) Pipeline

1. **Document Ingestion**
   - 80+ PKD research papers (PDFs) are processed and chunked into semantic segments
   - Each chunk is converted into embeddings using OpenAI's `text-embedding-3-small` model
   - Embeddings are stored in ChromaDB vector database for fast similarity search

2. **Query Processing**
   - User questions are analyzed by an Adaptive Agent that determines the optimal reasoning strategy
   - Questions are embedded using the same model for semantic similarity matching
   - Top-K most relevant chunks are retrieved from the vector database

3. **Answer Generation**
   - Retrieved context is passed to OpenAI's `gpt-4o` model along with the user's question
   - The model generates accurate, evidence-based answers grounded in the research papers
   - Source citations are automatically extracted and displayed with each response

### 🚀 Advanced Features

#### 1. **Chain-of-Thought (CoT) Reasoning**
- Breaks complex questions into logical reasoning steps
- Provides transparent thought process for each answer
- Ideal for medical queries requiring multi-step analysis

#### 2. **Stepback Query Decomposition**
- Transforms narrow questions into broader, principle-based queries
- Retrieves more comprehensive background knowledge
- Better handles questions that lack sufficient direct context

#### 3. **Adaptive Agent Selector**
- Automatically analyzes each question's complexity and type
- Dynamically selects the best reasoning mode (Standard, CoT, or Stepback)
- Optimizes response quality while minimizing latency

---

## 🛠️ Technology Stack

### Frontend
- **Flutter Web** - Cross-platform UI framework
- **Dart** - Programming language
- **Firebase Hosting** - Web deployment

### Backend
- **FastAPI** (Python) - High-performance REST API
- **OpenAI API** - GPT-4o for generation, text-embedding-3-small for embeddings
- **ChromaDB** - Vector database for semantic search
- **Google Cloud Run** - Serverless backend deployment

### Key Libraries
- `uvicorn` - ASGI server
- `langchain` - LLM orchestration utilities
- `pdfplumber` - PDF text extraction
- `httpx` - Async HTTP client

---

## 📊 Knowledge Base

The chatbot is trained on a curated collection of 80+ peer-reviewed research papers covering:
- PKD pathophysiology and genetics
- Treatment options and clinical trials
- Dietary and lifestyle management
- Emerging therapies and research

All responses include citations to source papers for verification and further reading.

---

## 🔧 Local Development

### Backend Setup
```bash
# Create and activate virtual environment
python3 -m venv backend_env
source backend_env/bin/activate

# Install dependencies
pip install -r backend/app/requirements_openai.txt

# Create .env file with OpenAI API key
echo "OPENAI_API_KEY=your_key_here" > backend/.env

# Start backend server
cd backend/app
uvicorn main_openai:app --host 0.0.0.0 --port 8000 --reload
```

### Frontend Setup
```bash
# Get Flutter dependencies
flutter pub get

# Run web app with local backend
flutter run -d chrome --dart-define=BACKEND_BASE_URL=http://localhost:8000
```

---

## 📝 License

Research papers are copyrighted by their respective publishers and not included in this repository.