import streamlit as st
import os

# LangChain components for document processing and QA
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain_huggingface import HuggingFaceEmbeddings, ChatOllama # Updated imports for local models
from langchain_huggingface import HuggingFaceEmbeddings # For the embeddings
from langchain_ollama import ChatOllama # For the Ollama chat model
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# --- Configuration ---
DOCUMENTS_FOLDER = "./my_documents"
OLLAMA_MODEL = "llama3" # Ensure you have downloaded this model via `ollama run llama3`
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2" # A good, small, fast embedding model
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TEMPERATURE = 0.7 # Controls creativity of LLM (0.0 for factual, higher for more creative)

# --- Streamlit UI ---
st.set_page_config(page_title="My Local Document Chatbot", layout="wide")
st.title("📚 My Local Document Chatbot (Open Source)")
st.markdown(f"""
Ask me questions about the documents in your `{DOCUMENTS_FOLDER}` folder!
Using **Ollama ({OLLAMA_MODEL})** for answers and **Hugging Face ({EMBEDDING_MODEL_NAME})** for document understanding.
""")

# --- Check Ollama Server Status ---
def check_ollama_status():
    try:
        # Attempt to connect to Ollama (default port is 11434)
        # This is a simple check, a more robust check would involve `ollama list`
        import requests
        requests.get("http://localhost:11434/api/tags", timeout=1)
        return True
    except requests.exceptions.ConnectionError:
        return False
    except Exception as e:
        st.error(f"Error checking Ollama status: {e}")
        return False

if not check_ollama_status():
    st.error("Ollama server is not running or not accessible. Please ensure Ollama is installed and running.")
    st.markdown("You can download Ollama from [ollama.com/download](https://ollama.com/download).")
    st.markdown(f"After installing, run `ollama run {OLLAMA_MODEL}` in your terminal to download the model and start the server.")
    st.stop() # Stop the Streamlit app if Ollama isn't running

# --- Function to Load and Process Documents ---
@st.cache_resource # Cache this function to run only once
def load_and_process_documents(folder_path, embedding_model_name):
    st.info("Loading and processing documents... This might take a moment.")
    try:
        # 1. Load Documents
        loader_mapping = {
            ".txt": TextLoader,
            ".pdf": PyPDFLoader,
        }

        all_documents = []
        for ext, loader_class in loader_mapping.items():
            loader = DirectoryLoader(
                folder_path,
                glob=f"**/*{ext}",
                loader_cls=loader_class,
                recursive=True,
                silent_errors=True
            )
            try:
                loaded_docs = loader.load()
                all_documents.extend(loaded_docs)
                st.success(f"Loaded {len(loaded_docs)} '{ext}' documents.")
            except Exception as e:
                st.warning(f"Could not load '{ext}' files: {e}")

        if not all_documents:
            st.error(f"No documents found or loaded from '{folder_path}'. Please check the folder and file types.")
            return None, None

        # 2. Split Documents into Chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            is_separator_regex=False,
        )
        chunks = text_splitter.split_documents(all_documents)
        st.info(f"Split documents into {len(chunks)} text chunks.")

        # 3. Create Embeddings and Store in Vector Store
        # Note: HuggingFaceEmbeddings will download the model the first time
        with st.spinner(f"Downloading and initializing embedding model ({embedding_model_name})..."):
            embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
        st.success("Embedding model initialized.")

        with st.spinner("Creating document embeddings and storing in FAISS..."):
            vector_store = FAISS.from_documents(chunks, embeddings)
        st.success("Document embeddings created and stored in FAISS vector store.")

        return vector_store, embeddings

    except Exception as e:
        st.error(f"An error occurred during document processing: {e}")
        return None, None

# --- Main Application Logic ---
if not os.path.exists(DOCUMENTS_FOLDER):
    st.error(f"The documents folder '{DOCUMENTS_FOLDER}' does not exist.")
    st.info("Please create a folder named 'my_documents' in the same directory as this script and put your files inside.")
else:
    # Load and process documents only once when the app starts
    vector_store, embeddings = load_and_process_documents(DOCUMENTS_FOLDER, EMBEDDING_MODEL_NAME)

    if vector_store:
        # Initialize the LLM via Ollama
        llm = ChatOllama(model=OLLAMA_MODEL, temperature=TEMPERATURE)

        # Custom prompt template for better control over the LLM's answer
        custom_prompt_template = """Use the following pieces of context from the documents to answer the user's question.
        If you don't know the answer based on the provided context, just say that you don't know.
        Do not try to make up an answer.
        ----------------
        Context: {context}
        Question: {question}
        Answer:
        """
        CUSTOM_PROMPT = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

        # Create the Retrieval QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(),
            return_source_documents=True,
            chain_type_kwargs={"prompt": CUSTOM_PROMPT}
        )

        # User input for the question
        user_question = st.text_input("What would you like to know about your documents?", key="user_input")

        if user_question:
            with st.spinner("Searching and generating answer..."):
                try:
                    result = qa_chain.invoke({"query": user_question})
                    st.subheader("Answer:")
                    st.write(result["result"])

                    if result.get("source_documents"):
                        st.subheader("Source Documents:")
                        for i, doc in enumerate(result["source_documents"]):
                            source_name = doc.metadata.get('source', 'Unknown Source')
                            page_label = doc.metadata.get('page', 'N/A')
                            st.markdown(f"- **Source:** `{os.path.basename(source_name)}` (Page: {page_label})")

                except Exception as e:
                    st.error(f"An error occurred while getting the answer: {e}")
                    st.warning("Please ensure the Ollama server is running and the model is downloaded correctly.")

    else:
        st.warning("Chatbot is not ready. Please ensure documents are present and processed successfully.")