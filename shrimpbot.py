import openai
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_models import ChatOpenAI
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import (
    StreamlitChatMessageHistory,
)

import shrimp_helper
from key_helper import check_openai_key
import streamlit as st

# Add input for OpenAI API key
with st.sidebar:
    target_word_input = st.text_input(
        "Text input for AI to use",
        "Shrimp",
        key="placeholder",
        max_chars=15
    )
    ai_mode_selection = st.radio(
        "Set word replace mode 👉",
        key="ai_mode",
        options=["Partial Shrimp Mode", "Full Shrimp Mode", "Normal"],
    )
    st.info("Full Shrimp Mode: Every word in the AI's response is replaced with your chosen keyword.")
    st.info("Partial Shrimp Mode: Some part of the AI's response are replaced with your chosen keyword.")
    openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
    "[Inspired by this post](https://www.facebook.com/groups/cursedaiwtf/posts/1395288517746294)"


st.title("Shrimp Transformer")
st.caption("")

# Set up memory
msgs = StreamlitChatMessageHistory(key="history")

check_openai_key(openai_api_key)

# Set up LLMs
llm_shrimp = ChatOpenAI(temperature=0.8, model="gpt-3.5-turbo-0125", request_timeout=30)
if "llm_shrimp_memory" not in st.session_state:
    llm_shrimp_memory = ConversationBufferMemory()
    st.session_state.llm_shrimp_memory = llm_shrimp_memory
else:
    llm_shrimp_memory = st.session_state.llm_shrimp_memory

# Render current messages from StreamlitChatMessageHistory
for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

if len(msgs.messages) == 0:
    init_msg = "Input something to start a conversation"
    st.chat_message("system").write(init_msg)

llm_shrimp_conver_chain = shrimp_helper.shrimpify_chat_prompt | llm_shrimp
llm_shrimp_conver_chain_with_history = RunnableWithMessageHistory(
    llm_shrimp_conver_chain,
    lambda session_id: msgs,
    input_messages_key="question",
    history_messages_key="history",
)

def generate_conversation(latest_response, ai_mode_selection, st):
    config = {"configurable": {"session_id": "any"}}
    ai_response = llm_shrimp_conver_chain_with_history.invoke({
        "mode": ai_mode_selection,
        "target_word": target_word_input,
        "question": latest_response}, config)
    ai_response = ai_response.content
    if ai_mode_selection == "Full Shrimp Mode":
        #print("FULL_SHRIMP_MODE")
        shrimpified_response = ' '.join(
            [target_word_input if target_word_input not in word else word for word in ai_response.split()])
        #print("shrimpified_response:", shrimpified_response)
        ai_response = shrimpified_response

    # Add the AI response to the conversation container
    #msgs.add_ai_message(ai_response)

    # Display the AI response in the chat interface
    st.chat_message("ai").write(ai_response)

    # The function should return the AI's response instead of the latest response from the user
    return ai_response


if prompt := st.chat_input():
    openai.api_key = openai_api_key
    #msgs.add_user_message(prompt)
    st.chat_message("user").write(prompt)
    latest_response = generate_conversation(prompt, ai_mode_selection, st)
    #print("ai_mode_selection:", ai_mode_selection)
