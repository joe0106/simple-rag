import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

# load .env
load_dotenv()

# load pdf
print("Loading PDF document...")
loader = PyPDFLoader("sample_doc.pdf")
document = loader.load()

#print(document[0].page_content)

# chunk up the doc content
print("Chunking text...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = text_splitter.split_documents(document)
#print(f"{0} chunk: ", chunks[0].page_content, f"\n{1} chunk: ", chunks[1].page_content)

# embed the chunks and map lists of numbers into Chroma
print("Creating vector database...")
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# setting up retriever and prompt template
retriever = vector_db.as_retriever(search_kwargs={"k": 2})
template = """
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use three sentences maximum and keep the answer concise.

Context: {context}

Question: {question}

Answer:
"""
prompt = PromptTemplate.from_template(template)

# init gemini model tier
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0)
# helper function to combinechunks inot single text block
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
# chain everything together using LCEL(Lang Chain Expression Language)
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
)

# # invoke chain with question
# user_question = "what days can i work from home?"
# print(f"\nQuestion: {user_question}")
# response = rag_chain.invoke(user_question)

print("\n--- PDF Chatbot Initialized ---")
print("Type 'exit' or 'quit' to stop ")

while True:
    user_question = input("\nYour Question: ")
    if user_question.lower() in ["exit", "quit"]:
        print("Shutting down chatbot. Goodbye!")
        break
    response = rag_chain.invoke(user_question)
    # filter the text value only, if response.content is a list
    if isinstance(response.content, list):
        clean_ans = response.content[0]['text']
    else:
        clean_ans = response.content
    print(f"\nAnswer: {clean_ans}")