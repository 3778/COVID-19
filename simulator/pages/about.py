import streamlit as st

from pages.utils import texts

def write():
    st.write("## Sobre o projeto")
    st.write(texts.ABOUT)
