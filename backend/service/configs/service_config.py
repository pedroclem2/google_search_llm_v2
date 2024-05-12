import os
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

#load google search credentials
def google_search_api_key():
    if "GOOGLE_API_KEY" in os.environ:
        return os.getenv("GOOGLE_API_KEY")
    elif "GOOGLE_API_KEY" in st.secrets:
        return st.secrets["GOOGLE_API_KEY"]
    else:
        raise KeyError("google search api key not found in envvariables.")

#load openai key
def get_openai_api_key():
    if "OPENAI_API_KEY" in os.environ:
        return os.getenv("OPENAI_API_KEY")
    elif "OPENAI_API_KEY" in st.secrets:
        return st.secrets["OPENAI_API_KEY"]
    else:
        raise KeyError("openai api key not found in environment variables")
    
#load search engine cx
def get_search_engine_cx():
    if "SEARCH_ENGINE_CX" in os.environ:
        return os.getenv("SEARCH_ENGINE_CX")
    elif "SEARCH_ENGINE_CX" in st.secrets:
        return st.secrets["SEARCH_ENGINE_CX"]
    else:
        raise KeyError("search engine cx not found")

