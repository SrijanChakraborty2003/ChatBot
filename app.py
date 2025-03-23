from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
import torch
import streamlit as st
import os
import asyncio
from huggingface_hub import login

HF_TOKEN = st.secrets.get("HF_TOKEN") or os.getenv("HF_TOKEN")

if not HF_TOKEN:
    st.error("Hugging Face token not found. Make sure it's set in Streamlit secrets.")
else:
    login(HF_TOKEN)
# Load BlenderBot model and tokenizer
tokenizer = BlenderbotTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
model = BlenderbotForConditionalGeneration.from_pretrained("facebook/blenderbot-400M-distill").to("cpu")

# Streamlit UI setup
st.title("🤖 Chatbot with BlenderBot-400M")
st.write("Talk to the chatbot and get responses in real-time!")

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Ensure async event loop works correctly in Streamlit
def get_event_loop():
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.new_event_loop()

loop = get_event_loop()

# User input
user_input = st.text_input("You:", "")
if user_input:
    st.session_state.chat_history.append(user_input)
    
    # Prepare input text
    input_text = " \n".join(st.session_state.chat_history[-5:])  # Limit chat history to last 5 messages
    inputs = tokenizer(input_text, return_tensors="pt").to("cpu")
    
    # Generate response
    with torch.no_grad():
        output = model.generate(
            **inputs, 
            max_new_tokens=50,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.2,
            do_sample=True
        )
    
    # Decode response
    bot_response = tokenizer.decode(output[0], skip_special_tokens=True)
    st.session_state.chat_history.append(bot_response)
    
    # Display chat history
    for i, msg in enumerate(st.session_state.chat_history[-10:]):
        if i % 2 == 0:
            st.text_area("You:", msg, height=50, disabled=True)
        else:
            st.text_area("Chatbot:", msg, height=50, disabled=True)
    
    # Keep chat history manageable
    st.session_state.chat_history = st.session_state.chat_history[-10:]
