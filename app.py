import streamlit as st
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. UI Layout
st.set_page_config(page_title="Ontario Edu RAG", page_icon="📚")
st.title("📚 Ontario Educational Policy QA System")
st.write("Ask questions based natively on the Ontario Curriculum documentation.")

# 2. Secure your API key via Streamlit Secrets
openai_api_key = st.secrets["OPENAI_API_KEY"]

# 3. Load your pre-built database (Make sure chroma_db folder is in your GitHub repo)
@st.cache_resource # Keeps the app fast by loading the database only once
def load_db():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=openai_api_key)
    return Chroma(persist_directory="./chroma_db", embedding_function=embeddings, collection_name="ontario_education_docs")

db = load_db()

# Initialize LLM & Chain
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=openai_api_key)
prompt_template = PromptTemplate.from_template("Answer based ONLY on context:\n\n{context}\n\nQuestion: {query}")
rag_chain = prompt_template | llm | StrOutputParser()

# 4. User Input Chat Box
if query_text := st.chat_input("Enter your question here..."):
    with st.chat_message("user"):
        st.write(query_text)
        
    with st.chat_message("assistant"):
        # Retrieve context
        results = db.similarity_search_with_relevance_scores(query_text, k=3)
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        
        # Run chain and stream/display response
        response = rag_chain.invoke({"context": context_text, "query": query_text})
        st.write(response)
