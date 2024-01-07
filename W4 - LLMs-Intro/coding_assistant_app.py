import streamlit as st
from coding_assistant import agent_chain  # Import your LangChain setup

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# Streamlit interface
st.title("LangChain Chat Assistant")

# Text input for user message
user_input = st.text_input("Your message:", key="user_input")

# Function to update chat history and interact with LangChain
def update_chat_history(user_message, agent_response):
    st.session_state['chat_history'].append({"user": user_message, "agent": agent_response})

    for index, chat in enumerate(st.session_state['chat_history'][::-1]):

        user_key = f"user_{index}"  # Unique key for the user's message
        assistant_key = f"assistant_{index}"  # Unique key for the assistant's message

        st.text_area("User:", value=chat["user"], height=100, disabled=True, key=user_key)
        st.text_area("Assistant:", value=chat["agent"], height=50, disabled=True, key=assistant_key)
# Handle user input
if st.button("Send"):
    if user_input:
        # Prepare the input for LangChain
        input_data = user_input
        chat_history = st.session_state['chat_history']

        # Interact with LangChain
        response = agent_chain.invoke({"input":input_data})

        agent_response = response.get("output", "Sorry, I couldn't process that.")

        # Update chat history
        update_chat_history(user_input, agent_response)
