import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Force CPU usage
device = torch.device("cpu")

# Load tokenizer separately with caching
@st.cache_resource()
def load_tokenizer():
    return AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# Load model with low CPU memory usage
@st.cache_resource()
def load_model():
    return AutoModelForCausalLM.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    ).to(device)

# Load tokenizer and model
tokenizer = load_tokenizer()
model = load_model()

# Streamlit UI
st.title("TinyLlama Chatbot")
st.write("Chat with TinyLlama - A lightweight AI assistant!")

# Initialize session state for chat messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for role, text in st.session_state.messages:
    if role in ["user", "assistant"]:
        st.chat_message(role).write(text)

# Chat input
user_input = st.chat_input("Type your message here...")

if user_input:
    st.chat_message("user").write(user_input)

    system_prompt = "You are a friendly AI assistant. Keep responses short and relevant. Answer conversationally and avoid generating code unless explicitly asked."
    full_prompt = f"{system_prompt}\nUser: {user_input}\nAI:"

    inputs = tokenizer(full_prompt, return_tensors="pt").to(device)

    def generate_response():
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=30,  # Reduce output length for faster response
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
