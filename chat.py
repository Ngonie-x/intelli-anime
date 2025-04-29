import streamlit as st
from llm_utils import get_model_response
import pandas as pd
import json

# --- Page Configuration (Optional) ---
st.set_page_config(page_title="LLM Chat", layout="wide")

st.title("Simple Chat with LLM 🤖")
st.caption("This chat can display text, dataframes, and image cards.")


def get_llm_response(user_prompt):
    """
    Calls the LLM with the user prompt and handles any exceptions.
    """
    try:
        response = get_model_response(user_prompt)
        return response
    except Exception as e:
        # Return a basic error response in the expected format
        return {"response_type": "text", "response_data": f"Error: {str(e)}"}


# --- Chat History Management ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Display Previous Messages ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # Check the type of content to display
        if message["role"] == "user":
            # User messages are always text
            st.write(message["content"])
        else:
            # For assistant, handle different response types
            content = message["content"]
            if content.response_type == "text":
                st.write(content.response_data)
            elif content.response_type == "dataframe":
                # Convert the DataFrameDict to a pandas DataFrame
                if hasattr(content.response_data, "data"):
                    # It's a DataFrameDict object with .data attribute
                    df = pd.DataFrame(json.loads(content.response_data.data))
                elif (
                    isinstance(content.response_data, dict)
                    and "data" in content.response_data
                ):
                    # It's a dict with "data" key
                    df = pd.DataFrame(content.response_data["data"])
                else:
                    # Try to convert directly
                    df = pd.DataFrame(json.loads(content.response_data))
                st.dataframe(df, use_container_width=True)
            elif content.response_type == "image":
                for card in content.response_data:
                    st.image(card.image_url, caption=card.title)
                    st.subheader(card.title)
                    st.write(card.description)
            else:
                # Fallback for unknown types
                st.write(f"Unsupported content type: {content['response_type']}")

# --- Handle User Input ---
if prompt := st.chat_input(
    "Ask the LLM something (e.g., 'show me anime with high ratings', 'recommend me an anime')"
):
    # 1. Add user message to chat history and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # 2. Get LLM response
    with st.spinner("LLM is thinking..."):
        llm_response = get_llm_response(prompt)["structured_response"]

    # 3. Add LLM response to chat history and display it
    st.session_state.messages.append({"role": "assistant", "content": llm_response})

    with st.chat_message("assistant"):
        # Check the type of content to display
        if llm_response.response_type == "text":
            st.write(llm_response.response_data)
        elif llm_response.response_type == "dataframe":
            # Convert the DataFrameDict to a pandas DataFrame
            df = pd.DataFrame(json.loads(llm_response.response_data))
            st.dataframe(df, use_container_width=True)
        elif llm_response.response_type == "image":
            for card in llm_response.response_data:
                # Handle both dictionary and ImageCard object formats
                if hasattr(card, "image_url"):
                    # It's an ImageCard object
                    st.image(card.image_url, caption=card.title)
                    st.subheader(card.title)
                    st.write(card.description)
                else:
                    # It's a dictionary
                    st.image(card["image_url"], caption=card["title"])
                    st.subheader(card["title"])
                    st.write(card["description"])
        else:
            # Fallback for unknown types
            st.error(f"Unsupported response type: {llm_response['response_type']}")

# --- Optional: Add a button to clear history ---
if st.sidebar.button("Clear Chat History"):
    st.session_state.messages = []
    st.rerun()  # Rerun the app to reflect the cleared state
