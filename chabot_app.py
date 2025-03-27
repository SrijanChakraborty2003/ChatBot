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
st.title("TinyLlama Chatbot 🤖")
st.write("Chat with TinyLlama - A lightweight AI assistant!")
if "messages" not in st.session_state:
    st.session_state.messages = []
for role, text in st.session_state.messages:
    st.chat_message(role).write(text)
user_input = st.chat_input("Type your message here...")
if user_input:
    st.chat_message("user").write(user_input)
    system_prompt = "You are a friendly AI assistant. Keep responses short and relevant. Answer conversationally and avoid generating code unless explicitly asked."
    full_prompt = f"{system_prompt}\nUser: {user_input}\nAI:"
    inputs = tokenizer(full_prompt, return_tensors="pt").to("cpu")
    with st.spinner("Thinking..."):
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        response = response.replace(full_prompt, "").split("\n")[0].strip()
    st.chat_message("assistant").write(response)
    st.session_state.messages.append(("user", user_input))
    st.session_state.messages.append(("assistant", response))
