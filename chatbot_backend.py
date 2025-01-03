import boto3
import streamlit as st
from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryBufferMemory
from langchain_aws import ChatBedrock

def demo_chatbot():

    # aws_access_key_id = st.secrets["AWS_ACCESS_KEY_ID"]
    # aws_secret_access_key = st.secrets["AWS_SECRET_ACCESS_KEY"]
    # region_name = st.secrets["AWS_REGION"]
        
    demo_llm = ChatBedrock(
        #boto3_session=session,
        aws_access_key_id='AKIA3LET5ZMHXM7PPGNO',
        aws_secret_access_key='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',
        region_name='us-east-1',
        model_id='amazon.nova-micro-v1:0',
        model_kwargs={
            "max_tokens": 100,
            "temperature": 0.3,
            "top_p": 0.9,
            "stop_sequences": ["\n\nHuman:"]
        }
    )
    return demo_llm
#2b Test out the LLM with Predict method instead use invoke method
   # return demo_llm.invoke(input_text)
#response=demo_chatbot(input_text="Hello Nova Micro, what is the capital of India?")
#print(response)

#3 Create a Function for  ConversationSummaryBufferMemory  (llm and max token limit)
def demo_memory():
    llm_data=demo_chatbot()
    memory=ConversationSummaryBufferMemory(llm=llm_data,max_token_limit=100)
    return memory

#4 Create a Function for Conversation Chain - Input text + Memory
def demo_conversation(input_text,memory):
    llm_chain_data=demo_chatbot()
    llm_conversation=ConversationChain(llm=llm_chain_data,memory=memory,verbose=True)

#5 Chat response using invoke (Prompt template)
    chat_reply=llm_conversation.invoke(input_text)
    return chat_reply['response']




