import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
from huggingface_hub import InferenceClient  
import pytesseract
from pdf2image import convert_from_path
import re

from htmlTemplates import css, bot_template, user_template

# Extract text from PDF (with OCR support)
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            if page.extract_text():
                text += page.extract_text()
        
        if not text.strip():  # If no text, use OCR
            images = convert_from_path(pdf)
            for image in images:
                text += pytesseract.image_to_string(image)

    return text.strip()

# Split extracted text into chunks
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=3000,
        chunk_overlap=400,
        length_function=len
    )
    return text_splitter.split_text(text)

# Convert text chunks into embeddings for retrieval
def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)

# Initialize conversational retrieval chain
def get_conversation_chain(vectorstore):
    client = InferenceClient("EleutherAI/gpt-neo-2.7B")  
    llm = HuggingFaceHub(
        repo_id="EleutherAI/gpt-neo-2.7B", 
        model_kwargs={"temperature": 0.7, "max_length": 512}, 
        client=client 
    )
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )

# Handle user input
def handle_userinput(user_question):
    if not st.session_state.conversation:
        st.warning("Please upload and process a PDF first.")
        return
    
    with st.spinner("Thinking... 💭"):
        user_question_lower = user_question.lower()
    
    extracted_response = ""

    # Check if the question relates to the PDF
    if not re.search(r'[a-zA-Z]', user_question_lower):
        extracted_response = "❗ Enter a valid question with meaningful words."

    elif len(user_question_lower) < 3 or user_question_lower in ["what is the pdf about", "what is this about"]:
        extracted_response = "❗ Please ask a more specific question related to the document content."

    elif not any(word in st.session_state.pdf_text.lower() for word in user_question_lower.split()):
        extracted_response = "⚠️ Nothing in PDF regarding this question."

    else:
        response = st.session_state.conversation({'question': user_question})
        chat_history = response.get('chat_history', [])
        model_response = chat_history[-1].content if chat_history else ""

        if "Helpful Answer:" in model_response:
            extracted_response = model_response.split("Helpful Answer:")[-1].strip()
        elif model_response.strip():
            extracted_response = model_response.strip()
        else:
            extracted_response = "⚠️ No relevant answer found in the PDF."

    # Store the latest Q&A at the top
    st.session_state.chat_history.insert(0, {"question": user_question, "answer": extracted_response})

    # Display each Q&A block (latest on top)
    for qa_pair in st.session_state.chat_history:
        with st.container():
            st.write(user_template.replace("{{MSG}}", qa_pair["question"]), unsafe_allow_html=True)
            st.write(bot_template.replace("{{MSG}}", qa_pair["answer"]), unsafe_allow_html=True)

# Main Streamlit app
def main():
    load_dotenv()
    st.set_page_config(page_title="📄 PDF Chat Assistant", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    # Initialize session state variables
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "pdf_text" not in st.session_state:
        st.session_state.pdf_text = ""

    st.header("📄 PDF Chat Assistant :books:")
    st.subheader("Upload PDFs and ask questions 📚")
    
    user_question = st.text_input("🔎 Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("📂 Upload Your PDFs")
        pdf_docs = st.file_uploader("Upload PDF files and click 'Process'", accept_multiple_files=True)
        
        if st.button("🚀Process"):
            
           if not pdf_docs:
               st.warning("❗ Please upload at least one file.")
           else:
               non_pdf_files = [file.name for file in pdf_docs if not file.name.lower().endswith(".pdf")]

           if non_pdf_files:
            st.error(f"❌ The following files are not PDFs:\n\n{', '.join(non_pdf_files)}\n\nThis system only supports PDF documents.")
           else:
            with st.spinner("Processing PDFs... ⏳"):
                raw_text = get_pdf_text(pdf_docs)

                if raw_text.strip():
                    text_chunks = get_text_chunks(raw_text)
                    vectorstore = get_vectorstore(text_chunks)
                    st.session_state.conversation = get_conversation_chain(vectorstore)
                    st.session_state.pdf_text = raw_text  # ✅ Store text for question validation
                    st.success("✅ PDFs processed successfully! You can now ask questions.")
                else:
                    st.warning("❌ No valid text found in the uploaded PDFs.")

if __name__ == '__main__':
    main()
