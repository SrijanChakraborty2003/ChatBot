import torch
import os
import asyncio
import streamlit as st
from huggingface_hub import login
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration

# Fix for Torch Initialization Issues
torch.classes.load_library = lambda _: None  # Disables unnecessary Torch class loading

# Load HF Token from Streamlit Secrets or Environment Variables
# HF_TOKEN = st.secrets.get("HF_TOKEN") or os.getenv("HF_TOKEN")

# if not HF_TOKEN:
#     st.error("Hugging Face token not found. Make sure it's set in Streamlit secrets.")
# else:
#     login(HF_TOKEN)
HF_TOKEN = "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"
login(HF_TOKEN)
# Load Model and Tokenizer
# Automatically select the available device
device = "cuda" if torch.cuda.is_available() else "cpu"
# Load model on the correct device
tokenizer = BlenderbotTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
model = BlenderbotForConditionalGeneration.from_pretrained("facebook/blenderbot-400M-distill").to(device)
#model = BlenderbotForConditionalGeneration.from_pretrained("facebook/blenderbot-400M-distill").to("cpu")

# Fix for Event Loop Issues
def get_event_loop():
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.new_event_loop()

loop = get_event_loop()

# Streamlit UI Setup
st.set_page_config(page_title="Chatbot", page_icon="💬", layout="centered")
st.markdown("""
    <style>
        .chat-container {
            max-width: 600px;
            margin: auto;
            padding: 20px;
            border-radius: 10px;
            background-color: #f9f9f9;
        }
        .user-msg {
            background-color: #0078ff;
            color: white;
            padding: 10px;
            border-radius: 10px;
            margin-bottom: 5px;
            text-align: right;
        }
        .bot-msg {
            background-color: #e0e0e0;
            padding: 10px;
            border-radius: 10px;
            margin-bottom: 5px;
            text-align: left;
        }
        .input-box {
            width: 100%;
            padding: 10px;
            border-radius: 5px;
            border: 1px solid #ccc;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
st.title("💬 AI Chatbot")
st.write("Talk to the chatbot and get responses in real-time!")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
for i, msg in enumerate(st.session_state.chat_history[-10:]):
    if i % 2 == 0:
        st.markdown(f"<div class='user-msg'>{msg}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='bot-msg'>{msg}</div>", unsafe_allow_html=True)

user_input = st.text_input("Type your message:", "", key="user_input")
if user_input:
    st.session_state.chat_history.append(user_input)
    
    # Prepare input text
    input_text = " \n".join(st.session_state.chat_history[-5:])
    inputs = tokenizer(input_text, return_tensors="pt").to("cpu")

    # Generate Response
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

    # Decode and Display Response
    bot_response = tokenizer.decode(output[0], skip_special_tokens=True)
    st.session_state.chat_history.append(bot_response)

    # Refresh UI
    st.rerun()

st.markdown("</div>", unsafe_allow_html=True)
