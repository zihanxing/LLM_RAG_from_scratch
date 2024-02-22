import os
import pandas as pd
import numpy as np
import pdfplumber
import sqlalchemy as sa
import openai
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from html_chatbot_template import css, bot_template, user_template


# Load the environment variables and the OpenAI API key
load_dotenv()
DB_string = os.getenv("DB_STRING")
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize the OpenAI client
client = OpenAI()

# Link to the CrateDB database
dburi = DB_string
engine = sa.create_engine(dburi, echo=False)

def extract_text(pdf_path):
    '''
    Function to extract the text from a PDF file.
    '''
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text


def get_chunks(text, chunk_size=300, overlap=50):
    """
    Function to get the chunks of text from the raw text
    
    Args:
    text (str): The raw text from the PDF file
    
    Returns:
    chunks (list): The list of chunks of text
    """
    # Ensure the overlap is not greater than the chunk size
    if overlap >= chunk_size:
        raise ValueError("Overlap must be less than the chunk size")
    
    # Initialize an empty list to hold the chunks of text
    chunks = []
    
    # Calculate the number of chunks
    num_chunks = len(text) // (chunk_size - overlap) + 1
    
    # Extract chunks of text
    for i in range(num_chunks):
        # Calculate start and end indices of the current chunk
        start_idx = i * (chunk_size - overlap)
        end_idx = start_idx + chunk_size
        # Append the current chunk to the chunks list
        chunks.append(text[start_idx:end_idx])
    
    # Return the list of chunks
    return chunks


def get_embeddings(text):
    '''
    Function to get the embeddings for a given text using the OpenAI API.

    Args:
    text (str): The text for which to get the embeddings

    Returns:
    embeddings (np.array): The embeddings for the text
    '''
    text = text.replace("\n", " ")
    return client.embeddings.create(input = [text], model="text-embedding-3-small").data[0].embedding


def get_embeddings_df(chunks):
    """
    Function to get the embeddings for each chunk of text and create a dataframe
    
    Args:
    chunks (list): The list of chunks of text
    
    Returns:
    df (pandas.DataFrame): The dataframe containing the text and embeddings of each chunk
    """
    # Initialize an empty list to hold the embeddings
    embeddings = []
    
    # Get the embeddings for each chunk
    for chunk in chunks:
        response = get_embeddings(chunk)
        embeddings.append(response)
    
    # Create a dataframe
    df = pd.DataFrame({
        "text": chunks,
        "embedding": embeddings
    })
    
    # Return the dataframe
    return df


def _upload_data_to_DB(df):
    '''
    Function to upload the data to the CrateDB database.
    '''
    with engine.connect() as connection:
        # Create database table.
        connection.execute(sa.text("CREATE TABLE IF NOT EXISTS text_data (text TEXT, embedding FLOAT_VECTOR(1536));"))
        # Write text and embeddings to CrateDB database.
        df.to_sql(name="text_data", con=connection, if_exists="append", index=False)
        connection.execute(sa.text("REFRESH TABLE text_data;"))


def get_response(my_question = "What is Japandi?"):
    '''
    Function to get the response to a user question using GPT-3.5, based on the context of the documents provided.

    Args:
    my_question (str): The user question

    Returns:
    response (str): The response to the user question
    '''

    query_embedding = get_embeddings(my_question)
    query_embedding = np.array(query_embedding).reshape(1, -1)

    with engine.connect() as connection:
        result = connection.execute(sa.text("SELECT text, embedding FROM text_data;"))
        rows = result.fetchall()
        text = [row[0] for row in rows]
        embeddings = [row[1] for row in rows]
        document_embeddings = np.array([np.array(x) for x in embeddings])

    # Calculate cosine similarity between query and document embeddings
    similarities = cosine_similarity(query_embedding, document_embeddings)

    sorted_doc_indices = np.argsort(-similarities[0])

    documents = []
    for i in range(4):
        documents.append(text[sorted_doc_indices[i]])

    # Concatenate the found documents into the context that will be provided in the system prompt
    context = '---\n'.join(doc for doc in documents)

    # Give instructions and context in the system prompt
    system_prompt = f"""
    You are a time series expert and get questions from the user covering the area of time series databases and time series use cases. 
    Please answer the users question in the language it was asked in. 
    Please only use the following context to answer the question, if you don't find the relevant information there, say "I don't know".

    Context: 
    {context}"""

    client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

    chat_completion = client.chat.completions.create(
        model="gpt-3.5-turbo", 
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": my_question}
        ]
    )

    # st.write(chat_completion.choices[0].message.content)
    st.write(bot_template.replace(
                "{{MSG}}", chat_completion.choices[0].message.content), unsafe_allow_html=True)
    

    chat_completion_naive = client.chat.completions.create(
        model="gpt-3.5-turbo", 
        messages=[
            {"role": "system", "content": "please answer my question"},
            {"role": "user", "content": my_question}
        ]
    )
    st.write(bot_template.replace(
                "{{MSG}}", 'WITHOUT RAG: '+chat_completion_naive.choices[0].message.content), unsafe_allow_html=True)


## Landing page UI
def UI_implmentation():
    
    # Upload the data to the database
    sample_text = extract_text("./Top-Interior-Design-Trends-2023-Feathr.com_.pdf")
    chunks = get_chunks(sample_text)
    df = get_embeddings_df(chunks)
    _upload_data_to_DB(df)

    # Set the page tab title
    st.set_page_config(page_title="RAG_from_scratch", page_icon="🤖", layout="wide")

    # Add the custom CSS to the UI
    st.write(css, unsafe_allow_html=True)

    # Initialize the session state variables to store the conversations and chat history
    if "conversations" not in st.session_state:
        st.session_state.conversations = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    # Set the page title
    st.header("Talk to a Interior Design Expert!")

    # Input text box for user query
    user_question = st.text_input("Ask me anything here...")

    # Check if the user has entered a query/prompt
    if user_question:
        # Call the function to generate the response
        get_response(user_question)



if __name__ == '__main__':
    UI_implmentation()


