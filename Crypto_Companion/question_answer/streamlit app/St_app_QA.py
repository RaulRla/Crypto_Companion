import streamlit as st
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

def answer_question(context, question, model, tokenizer):
    inputs = tokenizer(question, context, return_tensors="pt")
    outputs = model(**inputs)
    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits) + 1
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end]))
    return answer

model_name = "../t5-squad-finetuned/t5-squad-finetuned"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

st.title("Question Answering with Streamlit")

context = st.text_area("Enter context (or type 'exit' to quit):")
if context.lower() != 'exit':
    question = st.text_input("Ask a question:")
    if st.button("Get Answer"):
        answer = answer_question(context, question, model, tokenizer)
        st.write("Answer:", answer)
