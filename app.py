# app.py
import torch
import streamlit as st
from unsloth import FastLanguageModel

# ---------------- CONFIG ---------------- #
MODEL_NAME = "llama-3-8b-Instruct-bnb-4bit-medical"
MAX_SEQ_LEN = 2048
DTYPE = torch.float16
LOAD_IN_4BIT = True

# ---------------- LOAD MODEL ---------------- #
@st.cache_resource
def load_model():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LEN,
        dtype=DTYPE,
        load_in_4bit=LOAD_IN_4BIT,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer

model, tokenizer = load_model()

# ---------------- STREAMLIT UI ---------------- #
st.title("🩺 Medical LLaMA-3 Inference App")
st.write("Ask a medical question and get responses from your fine-tuned LLaMA-3 model.")

# Input box
user_question = st.text_area("Enter your medical question:", "")

if st.button("Generate Answer"):
    if user_question.strip() == "":
        st.warning("Please enter a question.")
    else:
        # Format input in the required format
        prompt = (
            "<|start_header_id|>system<|end_header_id|> "
            "Answer the question truthfully, you are a medical professional.<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|> "
            f"This is the question: {user_question}?<|eot_id|>"
        )

        # Tokenize
        inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

        # Generate response
        with st.spinner("Generating response..."):
            outputs = model.generate(**inputs, max_new_tokens=512, use_cache=True)

        # Decode response
        answer = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        st.subheader("Answer")
        st.write(answer)
