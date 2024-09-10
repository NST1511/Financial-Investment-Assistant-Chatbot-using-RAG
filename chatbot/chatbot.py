import streamlit as st
import os
import openai
from pydantic import BaseModel
from langchain import OpenAI, LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
import re

# ------------------------------------------------
# DEFINE FUNCTIONS

# Initialize OpenAI client
openai.api_key = os.environ.get('OPENAI_API_KEY')
MODEL = "gpt-4o-mini"
llm = ChatOpenAI(temperature=0, model=MODEL)

# Define structure for the answer
class Answer(BaseModel):
    answer_: str
    related_q1: str
    related_q2: str

sysmessage = """
You are a company information extractor and also an expert in Financial investment.
You will receive questions from users, and there are steps that you will follow to answer the questions:
First step (just think, don't write): you separate the question into sub-questions 
Second step (just think, don't write): Give detailed answers with numbers for evidence for each sub-question in a 2-level structure. Level 1 has 3 headings in a numbered list; each heading in level 1 is then explained in 3 bullet points. 
Third step (now write): Show the level 1 headings + their bullet points. Remember that you must always give numbers as evidence if possible.
Finally, generate 2 related questions that users might be interested in. 
Do not provide any information that is not in the context.
Do not give the answer in markdown format.
Important instruction: Answer 'No information available' if the information is not in the context.
"""

persist_directory = os.path.join(os.path.dirname(os.getcwd()), 'data', 'chroma', 'vietcap_reports')
embedding = OpenAIEmbeddings()
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
retriever = vectordb.as_retriever(search_kwargs={"k": 10})
template = sysmessage + "Based on the following context, answer the questions\n{context}\nand chat history\n{chat_history}\n### Question:\n{question}\n\n### Answer:"
prompt = PromptTemplate(template=template, input_variables=["question", "context", "chat_history"])

# Initialize session state for the memory, conversation history and the input field
if 'history' not in st.session_state:
    st.session_state.history = []

if 'current_question' not in st.session_state:
    st.session_state.current_question = ""

if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    return_source_documents=False,
    verbose=True,
    memory=st.session_state.memory,
    combine_docs_chain_kwargs={'prompt': prompt},
)

def remove_related_questions(text):
    return text.split("Related Questions:")[0]

def related_questions(conversation):
    related_questions = re.findall(r'Related Questions:\s*\n1\.\s*(.*)\n2\.\s*(.*)', conversation)
    if related_questions:   
        related_q1, related_q2 = related_questions[0]
    else :
        related_q1, related_q2 = None, None 
    return related_q1, related_q2


def get_answer_llm(question):
    answer_chain = qa_chain({"question": question, "chat_history": st.session_state.memory})

    answer = answer_chain['answer']
    answer = remove_related_questions(answer)
    related_q1, related_q2 = related_questions(answer_chain['answer'])

    # Append the new question and answer to history
    st.session_state.history.append({
        'question': question,
        'answer': answer,
        'related_q1': related_q1,
        'related_q2': related_q2
    })

def handle_related_question(related_question):
    get_answer_llm(related_question)

# ------------------------------------------------
# UI Setup
st.title("Financial Investment Assistant") 

# Define the submit function to handle user input
def submit():
    # Ensure that the user's input is captured
    st.session_state.current_question = st.session_state.widget
    st.session_state.widget = ''  # Reset the input widget
    # Get the answer and update the session state with the result
    get_answer_llm(st.session_state.current_question)


# Sidebar input form
with st.sidebar:
    st.text_area("Input your question:", key='widget', on_change=submit, height=150)

    if st.button('Get answer'):
        if st.session_state.widget.strip():  # Ensure the question is not empty
            submit()
        else:
            st.warning("Please input a valid question")


# Display conversation history in the main area
for idx, entry in enumerate(st.session_state.history):
    st.markdown(f"<div style='font-size:20px; font-weight:bold;'>USER:</div>", unsafe_allow_html=True)
    st.write(entry['question'])
    
    st.markdown(f"<div style='font-size:20px; font-weight:bold;'>ASSISTANT:</div>", unsafe_allow_html=True)
    st.write(entry['answer'])
    
    st.write("**Some follow-up questions you might be interested in:**")
    
    if entry['related_q1']:
        st.button(entry['related_q1'], on_click=handle_related_question, args=[entry['related_q1']], key=f"related_q1_{idx}")
    if entry['related_q2']:
        st.button(entry['related_q2'], on_click=handle_related_question, args=[entry['related_q2']], key=f"related_q2_{idx}")

    st.markdown(
        """
        <hr style="height:2px;border:none;color:#333;background-color:#333;" />
        """,
        unsafe_allow_html=True
    )
