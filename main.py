from flask import Flask, request, jsonify
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load OpenAI API Key from Railway env var
openai_api_key = os.environ.get("OPENAI_API_KEY")
llm = ChatOpenAI(openai_api_key=openai_api_key)

# Load Resume PDF
reader = PdfReader("ResumeDhanuj.pdf")
text = "\n".join([page.extract_text() for page in reader.pages])
chunks = CharacterTextSplitter(chunk_size=500, chunk_overlap=50).create_documents([text])
db = FAISS.from_documents(chunks, OpenAIEmbeddings(openai_api_key=openai_api_key))
qa = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())

@app.route("/")
def home():
    return "Resume Chatbot is live!"

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question", "")
    answer = qa.run(question)
    return jsonify({"answer": answer})
