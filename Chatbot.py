# Phase 1 import
import streamlit as st

# Phase2 import
import os
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Phase 3 import
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA


from dotenv import load_dotenv

load_dotenv()



st.title("Infinity Chatbot")

# Setup a session state variable to hold all the old messages
if 'messages' not in st.session_state:
    st.session_state.messages = []


# Display all the historical messages
for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

prompt = st.chat_input("Pass your prompt here")


if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({'role':'user','content':prompt})
    
    hf_sys_prompt = ChatPromptTemplate.from_template("""You are very smart at everything, you always give the best,
                                                        the most accurate and morst precise answers. Answer the following question: {user_prompt}.
                                                       """)
    
    
    llm = HuggingFaceEndpoint(
        repo_id="openai/gpt-oss-20b",
        task="text-generation"
    )
    
    model = ChatHuggingFace(llm=llm)
    
    # model="llama3-8b-8192"
    # groq_chat = ChatGroq(
    #     groq_api_key = os.environ.get("GROQ_API_KEY"),
    #     model_name=model
    # )

    chain = hf_sys_prompt | model | StrOutputParser()
    response = chain.invoke({"user_prompt":prompt}) 


    # response = "I am your Assistant"
    st.chat_message("assistant").markdown(response)
    st.session_state.messages.append({'role':'assistant','content':response})