import streamlit as st
import psutil

# Lazy import transformers to avoid conflicts
def load_transformers():
    global AutoModelForCausalLM, AutoTokenizer, torch
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    return AutoModelForCausalLM, AutoTokenizer, torch

# Load tokenizer function
@st.cache_resource()
def load_tokenizer():
    _, AutoTokenizer, _ = load_transformers()
    return AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# Load model function
@st.cache_resource()
def load_model():
    AutoModelForCausalLM, _, torch = load_transformers()
    return AutoModelForCausalLM.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        torch_dtype=torch.float32,
        device_map="cpu"
    )

# Streamlit UI Initialization
st.set_page_config(page_title="TinyLlama Chatbot", layout="centered")

st.title("TinyLlama Chatbot")
st.write("Chat with TinyLlama - A lightweight AI assistant!")

# Display available RAM
ram_available = psutil.virtual_memory().available / (1024 * 1024)  # Convert to MB
st.write(f"Available RAM: {ram_available:.2f} MB")

# Fix session state initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize model & tokenizer (inside Streamlit app)
try:
    tokenizer = load_tokenizer()
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")

# Display chat history
for role, text in st.session_state.messages:
    st.chat_message(role).write(text)

# Chat input
user_input = st.chat_input("Type your message here...")

if user_input:
    st.chat_message("user").write(user_input)

    # System prompt
    system_prompt = "You are a friendly AI assistant. Keep responses short and relevant. Answer conversationally and avoid generating code unless explicitly asked."
    full_prompt = f"{system_prompt}\nUser: {user_input}\nAI:"

    inputs = tokenizer(full_prompt, return_tensors="pt").to("cpu")

    # Generate response
    with st.spinner("Generating response..."):
        try:
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
        except Exception as e:
            st.error(f"Error generating response: {e}")
            response = "Sorry, I encountered an issue."

    st.chat_message("assistant").write(response)

    # Save chat history
    st.session_state.messages.append(("user", user_input))
    st.session_state.messages.append(("assistant", response))
