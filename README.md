📄 Creating a Large Language Model – PDF Chat Assistant

A conversational AI system that allows users to chat with their PDF documents using a Large Language Model (LLM).
Built with LangChain, FAISS, and GPT-Neo 2.7B on Streamlit.

🧠 Overview

The PDF Chat Assistant is an intelligent application that lets you upload PDFs and interact with them conversationally.
It uses OCR for scanned files, creates semantic embeddings from text, and retrieves relevant content using FAISS.
Responses are generated contextually using the GPT-Neo 2.7B model hosted on Hugging Face.

In simple terms — it’s like a mini ChatGPT that answers only from your uploaded PDF.

🚀 Features

✅ Upload multiple PDFs
✅ OCR support for scanned/image-based PDFs
✅ Semantic search using FAISS
✅ GPT-Neo 2.7B for contextual responses
✅ Conversation memory for multi-turn chat
✅ Modern and interactive Streamlit UI
✅ Error handling for invalid inputs or empty PDFs

⚙️ Tech Stack
Component	Technology
Language	Python
Frontend	Streamlit
LLM	GPT-Neo 2.7B (via Hugging Face Hub)
Embeddings	Sentence Transformers (all-MiniLM-L6-v2)
Vector Store	FAISS
Text Extraction	PyPDF2, pytesseract, pdf2image
Framework	LangChain
Environment Management	python-dotenv
🧩 Architecture
PDF Upload → Text Extraction (PyPDF2 / OCR)
             ↓
      Text Chunking (LangChain)
             ↓
   Embedding Generation (MiniLM)
             ↓
      Vector Storage (FAISS)
             ↓
  Query + Retrieval (RAG pipeline)
             ↓
   Response Generation (GPT-Neo)
             ↓
        Streamlit Chat UI

🛠️ Installation and Setup

Clone the repository

git clone https://github.com/<your-username>/PDF-Chat-Assistant.git
cd PDF-Chat-Assistant


Create and activate a virtual environment

python -m venv venv
source venv/bin/activate     # For Mac/Linux
venv\Scripts\activate        # For Windows


*Install dependencies

pip install -r requirements.txt


*Add your Hugging Face token

* Create a .env file in the root folder:

HUGGINGFACEHUB_API_TOKEN=your_huggingface_token_here

* Run the app

streamlit run mainapp.py

📚 How It Works

Upload one or more PDF files. --> The system extracts text using PyPDF2. --> If text is not readable, it uses pytesseract OCR. --> The text is split into smaller chunks (size: 3000, overlap: 400).
--> Each chunk is converted into embeddings and stored using FAISS.--> When you ask a question, the model: 1. Retrieves relevant chunks from FAISS.
                                                                                                           2. Sends context to GPT-Neo 2.7B for response generation.
--> The chatbot displays the contextual answer interactively.

🧮 Example Use Case
Query	Response
“What is the conclusion of this report?”	Returns conclusion text from the PDF
“Who are the authors?”	Extracts author names from the first page
“Explain the methodology used.”	Generates a concise summary from relevant sections

🏆 Publication
Published in IJCRT (International Journal of Creative Research Thoughts): Check it out: https://www.ijcrt.org/papers/IJCRT2505614.pdf
Title: Creating a Large Language Model – PDF Chat Assistant
Authors: Madhura Upadhye, Aditya Sarate, Utkarsha Chougule, Sourabh Gavandi, Mr Milind S. Vadagave
Guide: Mr. M. S. Vadagave
