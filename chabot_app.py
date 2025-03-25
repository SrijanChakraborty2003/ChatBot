import streamlit as st
import torch
import psutil
from transformers import AutoModelForCausalLM, AutoTokenizer

# Check available RAM before loading
ram_available = psutil.virtual_memory().available / (1024 * 1024)  # Convert to MB
st.write(f"Available RAM: {ram_available:.2f} MB")

# Load tokenizer
@st.cache_resource()
def load_tokenizer():
    return AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# Lazy model loading (avoids RAM overload)
def load_model():
    return AutoModelForCausalLM.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        torch_dtype=torch.float32,  # Ensures compatibility with CPU
        device_map="cpu"  # Forces CPU usage
    )

tokenizer = load_tokenizer()

st.title("TinyLlama Chatbot")
st.write("Chat with TinyLlama - A lightweight AI assistant!")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for role, text in st.session_state.messages:
    st.chat_message(role).write(text)

# Chat input
user_input = st.chat_input("Type your message here...")

if user_input:
    st.chat_message("user").write(user_input)

    system_prompt = "You are a friendly AI assistant. Keep responses short and relevant. Answer conversationally and avoid generating code unless explicitly asked."
    full_prompt = f"{system_prompt}\nUser: {user_input}\nAI:"

    inputs = tokenizer(full_prompt, return_tensors="pt").to("cpu")

    # Load model only when needed
    with st.spinner("Generating response..."):
        model = load_model()  # Load model only here (prevents Streamlit crashes)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=30,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    st.chat_message("assistant").write(response)

    # Save chat history
    st.session_state.messages.append(("user", user_input))
    st.session_state.messages.append(("assistant", response))
