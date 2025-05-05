# import the streamlit library for building the web app
import streamlit as st

# import langchain components needed for the qa system
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

# define function to process documents and generate answers
def generate_response(uploaded_file, openai_api_key, query_text):
    # check if a file was uploaded
    if uploaded_file is not None:
        # read the uploaded file's raw bytes
        raw_bytes = uploaded_file.read()
        # decode bytes to text if necessary
        text = raw_bytes.decode('utf-8') if isinstance(raw_bytes, bytes) else raw_bytes
        # create a list containing the document text
        documents = [text]

        # initialize text splitter with chunk size 1000
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        # split document into chunks
        texts = text_splitter.create_documents(documents)

        # create embeddings using openai api key
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        # create vector store from document chunks
        db = Chroma.from_documents(texts, embeddings)
        # create retriever interface for the vector store
        retriever = db.as_retriever()
        # initialize question-answering chain
        qa = RetrievalQA.from_chain_type(
            # specify the llm to use
            llm=OpenAI(openai_api_key=openai_api_key),
            # set chain type to 'stuff' (simple concatenation)
            chain_type='stuff',
            # connect the retriever
            retriever=retriever
        )
        # run the query and return response
        return qa.run(query_text)

# configure streamlit page settings
st.set_page_config(page_title='🦜🔗 ask the doc app')
# display app title
st.title('🦜🔗 ask the doc app')

# create file uploader for txt files
uploaded_file = st.file_uploader('upload an article', type='txt')
# create text input for questions (disabled until file upload)
query_text = st.text_input('enter your question:', placeholder='please provide a short summary.', disabled=not uploaded_file)

# initialize empty result list
result = []
# create form for api key submission
with st.form('myform', clear_on_submit=True):
    # create password input for openai api key
    openai_api_key = st.text_input('openai api key', type='password', disabled=not (uploaded_file and query_text))
    # create submit button (disabled until ready)
    submitted = st.form_submit_button('submit', disabled=not(uploaded_file and query_text))
    # check if submitted with valid api key
    if submitted and openai_api_key.startswith('sk-'):
        # show loading spinner
        with st.spinner('calculating...'):
            # generate response using uploaded data
            response = generate_response(uploaded_file, openai_api_key, query_text)
            # store response in result list
            result.append(response)
            # delete api key from memory
            del openai_api_key

# display result if available
if len(result):
    st.info(response)
