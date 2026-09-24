import os
import json
import chromadb
from dotenv import load_dotenv
from groq import Groq
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# ==========================================
# 🗄️ LOCAL VECTOR STORAGE PIPELINE
# ==========================================
def initialize_vector_database():
    """Reads corporate policy documentation, chunks it, and vectorizes it locally."""
    # 1. Read raw text artifact data
    if not os.path.exists("sample_policy.txt"):
        print("❌ System Error: 'sample_policy.txt' missing from project directory.")
        return None
        
    with open("sample_policy.txt", "r", encoding="utf-8") as file:
        raw_document_text = file.read()
        
    # 2. Chunking: Split long text data into digestible structural slices
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=30)
    text_chunks = text_splitter.split_text(raw_document_text)
    
    # 3. Setup a free, local SentenceTransformer embedding generator tool
    local_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    
    # 4. Instantiate a client database container in-memory
    chroma_client = chromadb.Client()
    
    # Re-create collection block pristine variables
    try:
        chroma_client.delete_collection(name="policy_knowledge_base")
    except Exception:
        pass
        
    collection = chroma_client.create_collection(
        name="policy_knowledge_base", 
        embedding_function=local_ef
    )
    
    # 5. Populate vectors mapping ids onto documents strings
    chunk_ids = [f"id_{i}" for i in range(len(text_chunks))]
    collection.add(documents=text_chunks, ids=chunk_ids)
    
    print(f"📢 [RAG Vector Setup]: Processed document successfully into {len(text_chunks)} local text vectors.")
    return collection

# ==========================================
# 🛠️ THE AGENT TOOLS: RAG VECTOR SEARCH
# ==========================================
def run_vector_search(query_string: str) -> str:
    """Queries the local ChromaDB database to retrieve the most contextually relevant document chunks."""
    try:
        # Reference our runtime collection memory directly
        results = active_collection.query(query_texts=[query_string], n_results=2)
        
        # Flatten the retrieved matching texts into a clean payload array string
        matched_documents = results.get("documents", [[]])[0]
        if not matched_documents:
            return "No matching internal corporate policy texts found in vector index search."
            
        context_payload = "--- RELEVANT INTERNAL POLICY EXCERPTS RECORDED ---\n"
        for i, text in enumerate(matched_documents):
            context_payload += f"[{i+1}] {text}\n"
        return context_payload
    except Exception as e:
        return f"Vector Query Error: Search execution failed. Details: {str(e)}"

# Meta JSON schema linking vector search capability straight to the model configuration
RAG_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "run_vector_search",
            "description": "Queries the local ChromaDB corporate knowledge base to retrieve semantic text matches regarding business policies.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query_string": {"type": "string", "description": "The look-up topic query term string."}
                },
                "required": ["query_string"]
            }
        }
    }
]

# ==========================================
# 🧠 THE ENGINE: RAG ANALYTICS workflow
# ==========================================
def run_rag_assistant_agent():
    global active_collection
    # Vectorize and instantiate database framework records first
    active_collection = initialize_vector_database()
    
    if active_collection is None:
        return
        
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    model_name = "qwen/qwen3.8-27b"
    
    # 🎯 TARGET QUESTION (Not inside the model's pre-trained data)
    user_query = "What is the maximum monthly internet stipend allowed for remote workers?"
    
    print("\n=============================================")
    print("🤖 RETRIEVAL-AUGMENTED GENERATION AGENT 🤖")
    print("=============================================\n")
    print(f"❓ [User Prompt Request]: {user_query}\n")
    
    messages = [
        {
            "role": "system",
            "content": (
                "You are an AI HR Assistant. You do NOT guess or make up details.\n"
                "When asked a policy question, you must first call run_vector_search to look up text blocks.\n"
                "Review the tool results. If the data contains the answer, summarize it cleanly for the user.\n"
                "If no metrics match, state that the information is unavailable in current corporate revisions."
            )
        },
        {"role": "user", "content": user_query}
    ]
    
    # Dynamic reasoning iteration loop
    for turn in range(3):
        print(f"🤖 [RAG Agent] Checking knowledge base semantic space (Turn {turn + 1}/3)...")
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            tools=RAG_TOOLS_SCHEMA,
            tool_choice="auto",
            max_tokens=500
        )
        
        #response_message = response.choices.message
        response_message = response.choices[0].message
        messages.append(response_message)
        
        if response_message.tool_calls:
            for tool_call in response_message.tool_calls:
                name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)
                
                print(f"⚡ [Agent Tool Action]: Querying ChromaDB local vector indexes...")
                print(f"🔍 [Semantic Search Query]: '{args.get('query_string')}'")
                
                if name == "run_vector_search":
                    output = run_vector_search(args.get("query_string"))
                else:
                    output = "Error: Unknown RAG search function vector schema tool."
                    
                print(f"📥 [Vector Match Data Returned]: Payload successfully appended to memory context window.")
                
                messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": name,
                    "content": output
                })
        else:
            print(f"\n📋 [Agent Authoritative Verified Answer]:\n{response_message.content}")
            break

if __name__ == "__main__":
    run_rag_assistant_agent()
