import streamlit as st
import torch
import asyncio
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the model and tokenizer with caching
@st.cache_resource()
def load_model():
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Ensure compatibility with CPU and avoid float16 issues
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float32,  # Changed from float16 to avoid CPU errors
        device_map="cpu"
    )
    
    return tokenizer, model

# Load model
tokenizer, model = load_model()

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
    # Display user message
    st.chat_message("user").write(user_input)

    # System prompt remains unchanged
    system_prompt = "You are a friendly AI assistant. Keep responses short and relevant. Answer conversationally and avoid generating code unless explicitly asked."
    full_prompt = f"{system_prompt}\nUser: {user_input}\nAI:"

    # Tokenize input
    inputs = tokenizer(full_prompt, return_tensors="pt").to("cpu")

    # Ensure compatibility with Streamlit's async event loop
    async def generate_response():
        with torch.no_grad():  # Disable gradient calculation for efficiency
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    # Check if we're in an async environment (to avoid Streamlit errors)
    if hasattr(asyncio, "run"):
        response = asyncio.run(generate_response())  # Run safely in Streamlit
    else:
        response = generate_response()  # Fallback for non-async environments

    # Display assistant response
    st.chat_message("assistant").write(response)

    # Update session state
    st.session_state.messages.append(("user", user_input))
    st.session_state.messages.append(("assistant", response))
