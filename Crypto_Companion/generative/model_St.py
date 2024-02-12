import streamlit as st
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

# model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('streamlit/gpt-2_medium/gpt2_medium_crypto')
model = GPT2LMHeadModel.from_pretrained('streamlit/gpt-2_medium/gpt2_medium_crypto')

# model to device
model.to(device)

# generate response
def generate_response(user_question):
    
    input_ids = tokenizer.encode(user_question, return_tensors='pt').to(device)

    # Generate response
    with torch.no_grad():
        output = model.generate(input_ids, max_length=250, num_beams=5, no_repeat_ngram_size=2, top_k=50, top_p=0.95, temperature=0.7)

    # Decode generated output and return as a string
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

# Streamlit app
st.title("GPT-2 Question Answering")

# input
user_question = st.text_area("Ask a question about cryptocurrency")

# Generate button
if st.button("Generate Response"):
    if user_question:
        # Generate response
        response = generate_response(user_question)
        st.text("Generated Response:")
        st.write(response)
    else:
        st.warning("Please enter a question.")
