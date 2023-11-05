import streamlit as st
import app as app
import textwrap
import os

os.environ["OPENAI_API_KEY"] = "paste your API key"
openai_api_key = os.environ["OPENAI_API_KEY"]

st.title("Youtube Assistant")


with st.sidebar:
    with st.form(key='my_form'):
        yt_url = st.sidebar.text_area(
            label = "Enter the url of the youtube video",
            max_chars=50
        )
        
        query = st.sidebar.text_area(
            label = "Ask me about the video?",
            max_chars=50,
            key = "query"
        )
        
        submit = st.form_submit_button(label = 'submit')
        
if query and yt_url:
    db = app.create_db_from_yt_vid_url(yt_url)
    response, docs = app.get_response_from_query(db, query)
    st.subheader("Answer:")
    st.text(textwrap.fill(response, width=85))
    
