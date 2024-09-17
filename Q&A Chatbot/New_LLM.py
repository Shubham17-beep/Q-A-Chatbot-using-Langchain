import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain



load_dotenv()

groq_api_key='*************************************';

st.title("Q&A")

llm=ChatGroq(groq_api_key=groq_api_key,
             model_name="Llama3-8b-8192")

prompt=ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions:{input}

"""
)

def vector_embedding():

    if "vectors" not in st.session_state:

        

        st.session_state.loader=PyPDFDirectoryLoader("C://Users//SHUBHAM MADHESIYA//Desktop//LLM//file") ## Data Ingestion
        st.session_state.docs=st.session_state.loader.load() ## Document Loading
        #st.write(st.session_state.docs)
        #st.write(st.session_state.docs[:10])
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200) ## Chunk Creation
        #st.write(st.session_state.text_splitter)
        #with open("vectors.txt", "w") as file: [file.write(" ".join(map(str, vector)) + "\n") for vector in st.session_state.docs]
        st.model_kwargs = {
                            "output_hidden_states": True
                            }
        st.model_name="sentence-transformers/all-mpnet-base-v2"
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[1:2]) #splitting
        #st.write(st.session_state.final_documents)
        st.session_state.model_kwargs = {'device': 'cpu'}
        print(st.session_state.final_documents)
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name=st.model_name,model_kwargs=st.session_state.model_kwargs,show_progress=True)
        st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings) #vector OpenAI embeddings
        #with open("vectors.txt", "w") as file: [file.write(" ".join(map(str, vector)) + "\n") for vector in st.session_state.vectors]
        #st.write(st.session_state.vectors)
        print(st.session_state.embeddings)
        print(st.session_state.vectors())



prompt1=st.text_input("Enter Your Question From Documents")


if st.button("Documents Embedding"):
    vector_embedding()
    print(st.session_state.vectors())
    st.write("Vector Store DB Is Ready")

import time



if prompt1:
    document_chain=create_stuff_documents_chain(llm,prompt)
    print(prompt1)
    retriever=st.session_state.vectors.as_retriever()
    retrieval_chain=create_retrieval_chain(retriever,document_chain)
    start=time.process_time()
    response=retrieval_chain.invoke({'input':prompt1})
    print("Response time :",time.process_time()-start)
    st.write(response['answer'])
    #st.write(response)
    
    # With a streamlit expander
    with st.expander("Document Similarity Search"):
        # Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")




