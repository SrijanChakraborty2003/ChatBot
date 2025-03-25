import streamlit as st
import torch
import psutil
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Check available RAM before loading
ram_available = psutil.virtual_memory().available / (1024 * 1024)  # Convert to MB
st.write(f"Available RAM: {ram_available:.2f} MB")

# Load tokenizer
@st.cache_resource()
def load_tokenizer():
    return AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# Load model with quantization
@st.cache_resource()
def load_model():
    quant_config = BitsAndBytesConfig(load_in_8bit=True)  # Use 8-bit for lower RAM usage
    return AutoModelForCausalLM.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        quantization_config=quant_config
    ).to("cpu")

# Lazy loading to prevent Streamlit timeout
if "model" not in st.session_state:
    with st.spinner("Loading model... This may take a while."):
        st.session_state.model = load_model()

tokenizer = load_tokenizer()
model = st.session_state.model

# Streamlit UI
st.title("TinyLlama Chatbot")
st.write("Chat with TinyLlama - A lightweight AI assistant!")

# Initialize session state for chat messages
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

    def generate_response():
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=30,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    # Generate and display response
    response = generate_response()
    st.chat_message("assistant").write(response)

    # Save conversation in session state
    st.session_state.messages.append(("user", user_input))
    st.session_state.messages.append(("assistant", response))
