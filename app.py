!pip install bitsandbytes-cuda117  # For CUDA 11.7
!pip install --upgrade pip
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install transformers langchain langchain-community huggingface_hub bitsandbytes faiss-cpu chromadb
!pip install -U langchain-huggingface
!pip install --upgrade langchain langchain-community
import langchain_community
from langchain.vectorstores import FAISS
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_community.llms import HuggingFacePipeline
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_huggingface import HuggingFaceEmbeddings 
symptom_checker_model_name = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(symptom_checker_model_name)
symptom_checker_model = AutoModelForCausalLM.from_pretrained(
    symptom_checker_model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)
mental_health_model = pipeline(
    "text-classification",
    model="facebook/bart-large-mnli",
    device=0 if torch.cuda.is_available() else -1
)embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
pip install chromadb langchain
from langchain_community.vectorstores import Chroma
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
pip install wikipedia
from langchain_community.document_loaders import WikipediaLoader

loader = WikipediaLoader(query="Medical conditions", lang="en", load_max_docs=5)
documents = loader.load()
from langchain_community.vectorstores import Chroma
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter

# Load embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Split documents into chunks
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# Store in ChromaDB
vector_db = Chroma.from_documents(
    docs, 
    embedding=embedding_model,  # ✅ Use 'embedding' instead of 'embedding_function'
    persist_directory="./medical_chroma_db"
)

# Save the database
vector_db.persist()
pip install -U langchain-chroma
from langchain_chroma import Chroma
from langchain_community.vectorstores import Chroma
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

# Load the embeddings model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ✅ Reload ChromaDB with correct parameter
vector_db = Chroma(
    persist_directory="./medical_chroma_db",
    embedding_function=embedding_model  # ✅ Correct argument
)

# ✅ Perform similarity search
query = "What are the symptoms of diabetes?"
retrieved_docs = vector_db.similarity_search(query, k=5)  # Retrieve top 5 documents

# ✅ Print retrieved documents
for i, doc in enumerate(retrieved_docs):
    print(f"\n🔹 Relevant Document {i+1}:\n{doc.page_content}")
    import torch
from transformers import BartForConditionalGeneration, BartTokenizer

# ✅ Determine the device
device = "cuda" if torch.cuda.is_available() else "cpu"

# ✅ Load the summarization model and tokenizer on the correct device
summarization_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn").to(device)
summarization_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

def retrieve_medical_info(query, vector_db, symptom_checker_model, tokenizer, mental_health_model, summarization_model, summarization_tokenizer):
    """Retrieve medical info efficiently, focusing on symptoms and refining mental health classification."""

    # ✅ Hybrid Search + Keyword Filtering
    retrieved_docs = vector_db.similarity_search_with_score(query, k=5)
    filtered_docs = [doc[0].page_content for doc in retrieved_docs if "symptom" in doc[0].page_content.lower() and "depression" in doc[0].page_content.lower()]
     if not filtered_docs:
        filtered_docs = [doc[0].page_content for doc in retrieved_docs[:2]]
    
    retrieved_text = "\n".join(filtered_docs)

    # ✅ Move tokenized inputs to the correct device
    inputs = summarization_tokenizer(retrieved_text, return_tensors="pt", truncation=True, max_length=1024).to(device)

    # ✅ Move the summarization model to the same device & run summarization
    summarization_model = summarization_model.to(device)
    summary_output = summarization_model.generate(**inputs, max_new_tokens=50)
    symptom_result = summarization_tokenizer.decode(summary_output[0], skip_special_tokens=True)

    # ✅ Better Mental Health Classification
    mental_health_results = mental_health_model(query)
    top_label = max(mental_health_results, key=lambda x: x['score'])  # Get highest-scoring label
label_explanations = {
        "positive": "The input suggests a positive or optimistic tone.",
        "negative": "The input suggests a negative or distressing tone.",
        "neutral": "The input is neutral and does not indicate strong emotional distress."
    }
    explanation = label_explanations.get(top_label["label"], "No explanation available.")

    return {
        "retrieved_documents": retrieved_text,
        "symptom_analysis": symptom_result,
        "mental_health_analysis": {
            "label": top_label["label"],
            "score": top_label["score"],
            "explanation": explanation
        }
    }

# ✅ Display Results
print("\n📚 Retrieved Documents:\n", response["retrieved_documents"])
print("\n🩺 Symptom Analysis:\n", response["symptom_analysis"])
print("\n🧠 Mental Health Analysis:\n", response["mental_health_analysis"])
# Streamlit UI
import streamlit as st
import requests

# Streamlit UI
st.set_page_config(page_title="Medical AI Assistant", layout="wide")
st.title("🩺 AI Medical Assistant")
st.subheader("Retrieve medical information instantly")

# User input
user_input = st.text_input("How can I assist you today?", "")

if st.button("Submit"):
    if user_input:
        with st.spinner("Processing your query..."):
            # Send request to backend
            response = requests.post(
                "https://YOUR_BACKEND_URL/api/query",  # Replace with your actual backend URL
                json={"query": user_input}
            )

            if response.status_code == 200:
                result = response.json().get("response", "No relevant information found.")
            else:
                result = "Error fetching response. Please try again."

        # Display response
        st.write("### Healthcare Assistant:")
        st.write(result)

    else:
        st.warning("⚠️ Please enter a query.")
