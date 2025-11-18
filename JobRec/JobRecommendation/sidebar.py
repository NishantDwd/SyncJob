import streamlit as st

# Set sidebar config
def sidebar():
    st.sidebar.title("About us")
    st.sidebar.subheader("")
    text_string_variable1=""
    url_string_variable1=""
    link = f'[{text_string_variable1}]({url_string_variable1})'
    st.sidebar.markdown(link, unsafe_allow_html=True)

    text_string_variable2=""
    url_string_variable2=""
    link = f'[{text_string_variable2}]({url_string_variable2})'
    st.sidebar.markdown(link, unsafe_allow_html=True) 


