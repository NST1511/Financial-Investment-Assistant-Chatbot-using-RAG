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

# ------------------------------------------------
# DEFINE FUNCTIONS

# Initialize OpenAI client
client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
openai.api_key = os.environ.get('OPENAI_API_KEY')
# Model specification
MODEL = "gpt-4o-mini"
# Initialize ChatOpenAI
llm = ChatOpenAI(temperature=0, model=MODEL)

# Define structure for the answer
class Answer(BaseModel):
    answer_: str
    related_q1: str
    related_q2: str

# Prompt template
sysmessage = """
You are a company information extractor and also an expert in Financial investment.
You will receive questions from users, and there are steps that you will follow to answer the questions:
First step (just think, don't write): you separate the question into sub-questions 
Second step (just think, don't write): Give detailed answers with numbers for evidence for each sub-question in a 2-level structure. Level 1 has 3 headings in a numbered list; each heading in level 1 is then explained in 3 bullet points. 
Third step (now write): Show the level 1 headings + their bullet points. Remember that you must always give numbers as evidence if possible.
Finally, generate 2 related questions that users might be interested in. 
Do not provide any information that is not in the context.
Do not give the answer in markdown format.
Important instruction: Answer 'I don't know' if the information is not in the context.
"""

# Specify the directory for persistent storage
persist_directory = os.path.join(os.path.dirname(os.getcwd()), 'data', 'chroma', 'vietcap_reports')
embedding = OpenAIEmbeddings()
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)

# Initialize the retriever
retriever = vectordb.as_retriever(search_kwargs={"k": 5})

# Prompt templates
template = sysmessage + "Based on the following context, answer the questions\n{context}\nand chat history\n{chat_history}\n### Question:\n{question}\n\n### Answer:"
prompt = PromptTemplate(template=template, input_variables=["question", "context", "chat_history"])
template_short = "History:\n{chat_history}\n### Question:\n{question}\n\n### Answer:"
prompt_short = PromptTemplate(template=template_short, input_variables=["question", "chat_history"])

# Initialize session state for memory
if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Initialize the QA chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    return_source_documents=False,
    verbose=True,
    memory=st.session_state.memory,
    combine_docs_chain_kwargs={'prompt': prompt},
    condense_question_prompt=prompt_short,
)

# function to remove the related questions from the answer
def remove_related_questions(text):
    text = text[:text.find("Related Questions:")]
    return text

# function to get the answer from the LLM
def get_answer_llm(question):
    # Pass chat history stored in memory
    answer_chain = qa_chain({"question": question, "chat_history": st.session_state.memory})

    response = client.beta.chat.completions.parse(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Read the provided text and extract the provided 2 related questions in JSON format."},
            {"role": "user", "content": answer_chain['answer']}
        ],
        temperature=0,
        max_tokens=1024,
        response_format=Answer,
    )
    answer = answer_chain['answer']
    answer = remove_related_questions(answer)
    related_q1 = response.choices[0].message.parsed.related_q1
    related_q2 = response.choices[0].message.parsed.related_q2

    # Append the new question and answer to history
    st.session_state.history.append({
        'question': question,
        'answer': answer,
        'related_q1': related_q1,
        'related_q2': related_q2
    })

# function to handle the related questions when clicked
def handle_related_question(related_question):
    get_answer_llm(related_question)

# ------------------------------------------------
# UI Setup
st.title("Financial Investment Assistant") 

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []

if 'current_question' not in st.session_state:
    st.session_state.current_question = ""

# Function to handle the submission of the question
def submit():
    st.session_state.current_question = st.session_state.widget
    st.session_state.widget = ''
    get_answer_llm(st.session_state.current_question)

# Place the input field and button inside the sidebar
with st.sidebar:
    st.text_area("Input your question:", key='widget', on_change=submit, height=150)

    question = st.session_state.get('current_question', '')
    askbutton = st.button('Get answer')

if askbutton:
    get_answer_llm(question)

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
