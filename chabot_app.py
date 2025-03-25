import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
@st.cache_resource()
def load_model():
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="cpu")
    return tokenizer, model
tokenizer, model = load_model()
st.title("TinyLlama Chatbot")
st.write("Chat with TinyLlama - A lightweight AI assistant!")
if "messages" not in st.session_state:
    st.session_state.messages = []
for message in st.session_state.messages:
    role, text = message
    if role == "user":
        st.chat_message("user").write(text)
    else:
        st.chat_message("assistant").write(text)
user_input = st.chat_input("Type your message here...")
if user_input:
    st.chat_message("user").write(user_input)
    system_prompt = "You are a friendly AI assistant. Keep responses short and relevant. Answer conversationally and avoid generating code unless explicitly asked."
    full_prompt = f"{system_prompt}\nUser: {user_input}\nAI:"
    inputs = tokenizer(full_prompt, return_tensors="pt").to("cpu")
    outputs = model.generate(
        **inputs, 
        max_new_tokens=50,
        do_sample=True, 
        temperature=0.7, 
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.replace(full_prompt, "").strip()
    response = response.split("\n")[0]
    st.chat_message("assistant").write(response)
    st.session_state.messages.append(("user", user_input))
    st.session_state.messages.append(("assistant", response))