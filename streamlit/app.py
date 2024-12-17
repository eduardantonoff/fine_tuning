import re
import torch
import streamlit as st
import pandas as pd
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# page settings
st.set_page_config(
    page_title="Pros & Cons Analyzer",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.title("pros & cons")

# model directory
MODEL_DIR = "./model"

@st.cache_resource
def load_model():
    """Load the tokenizer and model from the specified directory."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device

tokenizer, model, device = load_model()

# function to generate summary
def generate_summary(review, max_length=150, min_length=50):
    """
    Generate a summary for the given review.
    """
    input_text = f"Сделайте краткое резюме этого отзыва: {review}"
    inputs = tokenizer.encode(input_text, return_tensors="pt", truncation=True, max_length=512)
    inputs = inputs.to(device)

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            do_sample=True,
            max_length=max_length,
            min_length=min_length,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            top_p=0.9,
            temperature=0.7,
            top_k=50,
            early_stopping=True,
            pad_token_id=tokenizer.eos_token_id
        )

    generated = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return generated

# function to parse the summary into strengths and weaknesses
def parse_summary(summary_text):
    strengths = []
    weaknesses = []
    
    # regular expressions to find sections
    strengths_match = re.search(r"Сильные стороны[:\-]?\s*(.*?)(Слабые стороны|$)", summary_text, re.DOTALL | re.IGNORECASE)
    weaknesses_match = re.search(r"Слабые стороны[:\-]?\s*(.*)", summary_text, re.DOTALL | re.IGNORECASE)
    
    if strengths_match:
        strengths_text = strengths_match.group(1)
        strengths = re.findall(r'\d+\.\s*([^.\n]+)', strengths_text)
        # remove any trailing numbers
        strengths = [re.sub(r'\s*\d+$', '', item).strip() for item in strengths]
    
    if weaknesses_match:
        weaknesses_text = weaknesses_match.group(1)
        weaknesses = re.findall(r'\d+\.\s*([^.\n]+)', weaknesses_text)
        # remove any trailing numbers
        weaknesses = [re.sub(r'\s*\d+$', '', item).strip() for item in weaknesses]
    
    return strengths, weaknesses

# function to generate a DataFrame for pros and cons
def generate_pros_cons_table(strengths, weaknesses):
    # ensure both lists have the same length 
    max_len = max(len(strengths), len(weaknesses))
    strengths_extended = strengths + [""] * (max_len - len(strengths)) 
    weaknesses_extended = weaknesses + [""] * (max_len - len(weaknesses))

    # create a DataFrame
    data = {"Pros": strengths_extended, "Cons": weaknesses_extended}
    df = pd.DataFrame(data)
    return df

# function to aggregate pros and cons from multiple reviews
def aggregate_pros_cons(df_reviews):
    all_pros = []
    all_cons = []
    total_reviews = len(df_reviews)
    
    progress_bar = st.progress(0)
    
    for idx, review in enumerate(df_reviews, 1):
        summary = generate_summary(review)
        strengths, weaknesses = parse_summary(summary[0])
        all_pros.extend(strengths)
        all_cons.extend(weaknesses)
        progress_bar.progress(idx / total_reviews)
    
    progress_bar.empty()
    
    # aggregate counts
    pros_series = pd.Series(all_pros).value_counts().reset_index()
    pros_series.columns = ['Pros', 'Count']
    
    cons_series = pd.Series(all_cons).value_counts().reset_index()
    cons_series.columns = ['Cons', 'Count']
    
    return pros_series, cons_series

# create tabs for navigation
tabs = st.tabs(["Single Review", "Bulk Processing"])

with tabs[0]:
    # user input for single review
    st.caption("Enter your review here:")
    review = st.text_area(label="", height=200)
    
    if st.button("Submit"):
        if review.strip() == "":
            st.warning("Please enter a review to generate a summary.")
        else:
            with st.spinner("Generating summary..."):
                summaries = generate_summary(review)
            for idx, summary in enumerate(summaries, 1):
                strengths, weaknesses = parse_summary(summary)
                
                # generate & display pros and cons
                pros_cons_df = generate_pros_cons_table(strengths, weaknesses)
                st.table(pros_cons_df)

with tabs[1]:    
    st.markdown("""
    Upload a CSV file containing reviews (`clean_text`) to extract and aggregate pros and cons.
    """)
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            if 'clean_text' not in df.columns:
                st.error("The uploaded CSV does not contain a 'clean_text' column.")
            else:
                st.success("File uploaded successfully!")
                st.write(f"Total reviews to process: {len(df)}")
                
                if st.button("Process Reviews"):
                    with st.spinner("Processing reviews... This may take a while depending on the number of reviews."):
                        pros_series, cons_series = aggregate_pros_cons(df['clean_text'].tolist())
                    
                    st.success("Processing completed!")
                    
                    st.subheader("Aggregated Pros")
                    st.table(pros_series)
                    
                    st.subheader("Aggregated Cons")
                    st.table(cons_series)
                    
        except Exception as e:
            st.error(f"An error occurred while processing the file: {e}")
