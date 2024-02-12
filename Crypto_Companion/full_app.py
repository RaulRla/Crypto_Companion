import streamlit as st
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, GPT2LMHeadModel, GPT2Tokenizer
import torch
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta



def answer_question(context, question, model, tokenizer):
    inputs = tokenizer(question, context, return_tensors="pt")
    outputs = model(**inputs)
    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits) + 1
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end]))
    return answer

def generate_response(user_question, model, tokenizer, device):
    
    input_ids = tokenizer.encode(user_question, return_tensors='pt').to(device)

    # Generate response
    with torch.no_grad():
        output = model.generate(input_ids, max_length=70, num_beams=5, no_repeat_ngram_size=2, top_k=50, top_p=0.95, temperature=0.7)

    # Decode and return as a string
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Question Answering t5
qa_model_name = "question_answer/t5-squad-finetuned/t5-squad-finetuned"
qa_tokenizer = AutoTokenizer.from_pretrained(qa_model_name)
qa_model = AutoModelForQuestionAnswering.from_pretrained(qa_model_name)

# Load GPT-2 
gpt2_model_path = "generative/streamlit/gpt-2_medium/gpt2_medium_crypto"
gpt2_tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_path)
gpt2_model = GPT2LMHeadModel.from_pretrained(gpt2_model_path)
gpt2_model.to(device)

# Streamlit app
st.title("Crypto Companion")

# Section for Question Answering Model
st.sidebar.title("Question Answer")
context = st.sidebar.text_area("Enter context (or type 'exit' to quit):")
if context.lower() != 'exit':
    question = st.sidebar.text_input("Ask a question:")
    if st.sidebar.button("Get Answer"):
        answer = answer_question(context, question, qa_model, qa_tokenizer)
        st.sidebar.write("Answer:", answer)

# Section for GPT-2
user_question_gpt2 = st.text_area("Ask a question about cryptocurrency")

# Generate button for GPT-2 
if st.button("Generate Response"):
    if user_question_gpt2:
        response_gpt2 = generate_response(user_question_gpt2, gpt2_model, gpt2_tokenizer, device)
        st.text("Generated Response:")
        st.write(response_gpt2)
    else:
        st.warning("Please enter a question.")

# Section for Cryptocurrency
st.sidebar.title("Crypto Companion Chart")
selected_crypto = st.sidebar.selectbox("Select Cryptocurrency", ["BTC-USD", "ETH-USD", "XRP-USD", "LTC-USD", "BCH-USD", "ADA-USD"])
start_date = st.sidebar.date_input("Select Start Date", value=None)
end_date = st.sidebar.date_input("Select End Date", value=datetime.today())

fetching_data = st.empty()

if selected_crypto:
    if start_date and end_date:
        df = yf.download(selected_crypto, start=start_date, end=end_date)
    else:
        df = yf.download(selected_crypto)

    if not df.empty:
        fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'])])
        st.write("Candlestick Chart:")
        st.plotly_chart(fig)