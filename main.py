import os
import uuid
from typing import List, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

load_dotenv()

app = FastAPI()

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- In-memory storage for sessions ---
sessions = {}

class SessionData:
    def __init__(self):
        self.vector_store = None
        self.filenames = []
        self.all_chunks = [] # Store all text chunks for rebuilding the index

# --- Pydantic Models for Request Bodies ---
class QueryRequest(BaseModel):
    question: str
    session_id: str

class DocumentRequest(BaseModel):
    filename: str
    session_id: str

class ClearSessionRequest(BaseModel):
    session_id: str

# --- Helper Functions ---

def get_text_chunks(documents):
    """Splits documents into manageable chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    all_chunks = []
    for doc in documents:
        chunks = text_splitter.split_documents([doc])
        all_chunks.extend(chunks)
    return all_chunks

def get_vector_store(text_chunks):
    """Creates or updates a FAISS vector store from text chunks."""
    if not text_chunks:
        return None
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)
        return vector_store
    except Exception as e:
        print(f"Error creating vector store: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create vector store: {e}")


def get_conversational_chain():
    """Creates the conversational chain with a custom prompt."""
    prompt_template = """
    You are a helpful AI assistant specializing in legal document analysis. Your primary goal is to simplify complex legal text and empower the user to make an informed decision.
    Make the summary as easliy understandable as possible and also try to make it as short as u can.
    Based on the provided context from the legal document(s), your task is to:

    - Provide a clear, concise, and easy-to-understand analysis of the user's question.
    - Avoid using legal jargon yourself.
    - If the context contains the answer, use it to formulate your response. If not, state that "The document(s) do not provide information on this topic."

    For specific user requests, follow these additional rules:

    - **Summary:** If asked for a summary, provide a high-level overview of the document's purpose and key clauses.
    - **Risks & Benefits:** If asked about risks, highlight clauses that could be unfavorable or require caution. If asked about benefits, highlight clauses that appear favorable. You must present these as factual points from the document, not as personal opinions.
    - **Decision-Making:** If a user asks for help making a decision, do not tell them what to do. Instead, present a balanced "Key Considerations" or "Pros and Cons" list based on the document's contents. For example, point out a clause that gives a party broad power versus a clause that limits a user's liability.
    - **Next Steps:** Suggest potential non-legal next steps for the user to consider, such as "You might want to discuss this clause with the other party" or "You may want to verify this information."

    Crucially, you must include this disclaimer at starting responce or the first answer of yours:
    > "Please note: I am an AI assistant and not a legal professional. This information is for educational purposes only and does not constitute legal advice. For proper legal guidance, you must consult with a qualified attorney."

    Context:\n {context}\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# --- API Endpoints ---

@app.post("/process-documents/")
async def process_documents_endpoint(files: List[UploadFile] = File(...), session_id: Optional[str] = Form(None)):
    """Processes uploaded documents, adds them to a session, and rebuilds the vector store."""
    if not session_id or session_id not in sessions:
        session_id = str(uuid.uuid4())
        sessions[session_id] = SessionData()
    
    session_data = sessions[session_id]
    
    new_docs = []
    processed_filenames = []

    for file in files:
        if file.filename in session_data.filenames:
            continue

        file_extension = os.path.splitext(file.filename)[1].lower()
        temp_file_path = f"temp_{uuid.uuid4()}_{file.filename}"

        try:
            with open(temp_file_path, "wb") as f:
                f.write(await file.read())

            if file_extension == ".pdf":
                loader = PyPDFLoader(temp_file_path)
            elif file_extension == ".txt":
                loader = TextLoader(temp_file_path)
            else:
                continue

            documents = loader.load()
            # Add original filename to metadata for later removal
            for doc in documents:
                doc.metadata["source_filename"] = file.filename
            
            new_docs.extend(documents)
            processed_filenames.append(file.filename)

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing file {file.filename}: {e}")
        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
    
    if not new_docs:
         return {"session_id": session_id, "filenames": session_data.filenames, "message": "No new valid documents to process."}

    # Add new chunks to the session and rebuild the entire vector store
    new_chunks = get_text_chunks(new_docs)
    session_data.all_chunks.extend(new_chunks)
    session_data.filenames.extend(processed_filenames)
    
    session_data.vector_store = get_vector_store(session_data.all_chunks)

    return {"session_id": session_id, "filenames": session_data.filenames}

@app.post("/remove-document/")
async def remove_document_endpoint(request: DocumentRequest):
    """Removes a document from a session and rebuilds the vector store."""
    session_id = request.session_id
    filename_to_remove = request.filename

    if not session_id or session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found.")
        
    session_data = sessions[session_id]
    
    if filename_to_remove not in session_data.filenames:
        raise HTTPException(status_code=404, detail="Filename not found in session.")

    # Filter out chunks from the document to be removed
    session_data.all_chunks = [
        chunk for chunk in session_data.all_chunks 
        if chunk.metadata.get("source_filename") != filename_to_remove
    ]
    session_data.filenames.remove(filename_to_remove)

    # Rebuild the vector store with the remaining chunks
    session_data.vector_store = get_vector_store(session_data.all_chunks)

    return {"session_id": session_id, "filenames": session_data.filenames}

@app.post("/query/")
async def handle_query(request: QueryRequest):
    """Handles user questions for a given session."""
    if request.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found.")

    session_data = sessions[request.session_id]
    if not session_data.vector_store:
         raise HTTPException(status_code=400, detail="No documents are currently active in this session. Please upload a document.")
    
    vector_store = session_data.vector_store
    
    try:
        docs = vector_store.similarity_search(request.question)
        chain = get_conversational_chain()
        response = chain.invoke({"input_documents": docs, "question": request.question}, return_only_outputs=True)
        return {"answer": response["output_text"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during query processing: {e}")

@app.post("/clear-session/")
async def clear_session(request: ClearSessionRequest):
    """Clears a session from memory."""
    if request.session_id in sessions:
        del sessions[request.session_id]
        return {"message": f"Session {request.session_id} cleared."}
    raise HTTPException(status_code=404, detail="Session not found.")

@app.get("/", response_class=HTMLResponse)
async def get_frontend():
    """Serves the main HTML file."""
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="index.html not found.")

