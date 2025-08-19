import boto3
import json
import os
import sys
import streamlit as st

## We will be using Titan Embedding Model to generate Embedding
from langchain_aws import BedrockEmbeddings
from langchain_community.llms import Bedrock

## Data ingestion import
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader

## Vector Embedding And Vector Store import
from langchain_community.vectorstores import FAISS

## LLM Models import
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

## Bedrock Clients Implementation
bedrock=boto3.client(service_name="bedrock-runtime")
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock)

## Data Ingestion Implementation
def ingest_data():
    pdf_loader = PyPDFDirectoryLoader("data")
    documents = pdf_loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    docs = text_splitter.split_documents(documents)
    return docs

## Vector Embedding And Vector Store Implementation
def create_vector_store(docs):
    vector_store_faiss = FAISS.from_documents(
        docs, 
        bedrock_embeddings
    )
    vector_store_faiss.save_local("faiss_index")


def get_claude_llm():
    ## create the Anthropic model
    llm=Bedrock(model_id="anthropic.claude-3-sonnet-20240229-v1:0",client=bedrock,
                model_kwargs={'max_tokens':512})
    return llm

def get_llama2_llm():
    ## create the LLaMA2 model
    llm=Bedrock(model_id="meta.llama3-8b-instruct-v1:0",client=bedrock,
                model_kwargs={'max_gen_len':512})
    return llm

prompt_template="""
Human: Use the following pieces of context to provide a 
concise answer to the question at the end but usse atleast summarize with 
250 words with detailed explaantions. If you don't know the answer, 
just say that you don't know, don't try to make up an answer.
<context>
{context}
</context

Question: {question}

Assistant:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

def get_response_llm(llm, vector_store_faiss, query):
    qa=RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store_faiss.as_retriever(
            search_type="similarity", search_kwargs={"k": 3}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    answer = qa({"query": query})
    return answer['result']

def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using AWS Bedrock")

    user_question = st.text_input("Ask a question about the PDF Files:")

    with st.sidebar:
        st.title("Update Or Create Vector Store:")

        if st.button("Vectors Update"):
            docs = ingest_data()
            create_vector_store(docs)
            st.success("Vector store updated successfully!")

    if st.button("Claude Output"):
        with st.spinner("Generating response..."):
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
            llm = get_claude_llm()

            response = get_response_llm(llm, faiss_index, user_question)
            st.write("Response:")
            st.write(response)
            st.success("Done")

    if st.button("LLaMA2 Output"):
        with st.spinner("Generating response..."):
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
            llm = get_llama2_llm()

            response = get_response_llm(llm, faiss_index, user_question)
            st.write("Response:")
            st.write(response)
            st.success("Done")

if __name__ == "__main__":
    main()