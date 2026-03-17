import streamlit as st

st.title("AI Semantic Search")

query = st.text_input("Enter your search query:")

if query:
    st.write("You searched for:", query)
