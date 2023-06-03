import os
from langchain.document_loaders import TextLoader
from  langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import pinecone
from langchain import VectorDBQA, OpenAI

pinecone.init(api_key=os.environ.get('PINECONE_API_KEY'), environment="northamerica-northeast1-gcp")

if __name__ == '__main__':
    print ('Hello Vector Store!')
    loader = TextLoader("C:\\Users\\malek\\Documents\\python small programs\\vector-db-demo\\mediumblogs\\mediumblog1.txt", encoding='utf-8', autodetect_encoding=True)
    document = loader.load()
    
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(len(texts))

    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get('OPENAI_API_KEY'))
    docsearch = Pinecone.from_documents(texts, embeddings, index_name="mediumblogs")

    qa = VectorDBQA.from_chain_type(llm=OpenAI(), chain_type="stuff", vectorstore=docsearch, return_source_documents = True)

    query = "What is a vector DB? give me a 15 word answer for a beginner"
    result = qa({"query": query})
    print(result)