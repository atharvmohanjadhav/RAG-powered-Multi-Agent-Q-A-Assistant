import os
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.retrievers import PineconeHybridSearchRetriever
# from langchain.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone,ServerlessSpec
from pinecone_text.sparse import BM25Encoder

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "rag-example-index"

# Load PDF
def extract_text_from_doc(path):
    loader = PyPDFLoader(path)
    documents = loader.load()
    text = []
    for i in documents:
        text.append(i.page_content)
    return text

text = extract_text_from_doc("into_to_AI.pdf")


# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME, 
        dimension=384, 
        metric="dotproduct", 
        spec=ServerlessSpec(cloud="aws", region="us-east-1") 
    )

index = pc.Index(INDEX_NAME)
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

embed = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

bm2_encoder = BM25Encoder().default()
bm2_encoder.fit(text)   #apply TF-IDF values 
bm2_encoder.dump("bm2_values.json")

retriever = PineconeHybridSearchRetriever(embeddings=embed,sparse_encoder=bm2_encoder,index=index)
retriever.add_texts(texts=text)
info = retriever.invoke("what is AI?")
print(info)


