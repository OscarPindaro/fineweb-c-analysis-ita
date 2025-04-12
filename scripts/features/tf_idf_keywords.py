# ---
# jupyter:
#   jupytext:
#     formats: notebooks//ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# Add description here
#
# *Note:* You can open this file as a notebook (JupyterLab: right-click on it in the side bar -> Open With -> Notebook)


# %%
# Uncomment the next two lines to enable auto reloading for imported modules
# # %load_ext autoreload
# # %autoreload 2
# For more info, see:
# https://docs.ploomber.io/en/latest/user-guide/faq_index.html#auto-reloading-code-in-jupyter

# %% tags=["parameters"]
# If this task has dependencies, list them them here
# (e.g. upstream = ['some_task']), otherwise leave as None.
upstream = None

# This is a placeholder, leave it as None
product = None


# %% tags=["injected-parameters"]
# Parameters
n_keywords = 5
upstream = {"download-dataset": "/home/oscar/Progetti/fineweb-c-analysis-ita/products/get/dataset.parquet"}
product = {"nb": "/home/oscar/Progetti/fineweb-c-analysis-ita/products/features/tf_idf_keywords.ipynb", "dataset": "/home/oscar/Progetti/fineweb-c-analysis-ita/products/features/df_with_keywords.parquet"}


# %% [markdown]
# # Objective
# The objective is to extract important keywords for each sentence.
# I will use them to find high level features in another notebook, with the help of an LLM.
# I think that the "high quality" samples are mainly from teological and political blogs, and therefore a classifier would be heavily skewed towards this type of content.
# I have this hunch because I annotated roughly 400 samples and I noticed this pattern.

# %% [markdown]
# Load the dataset from hugging face. I'm personally interest in the italian split.

# %%
import pandas as pd

# %%
df = pd.read_parquet(upstream["download-dataset"])

# %% [markdown]
# ## Text cleaning
# Since i'm going to use tf-idf to find the keywords, I need to perform a cleaning step. 
# This is because tf-idf is very sensitive to the words being used in every document, and I want to exclude a priori irrelevant words such as articles, conjunctions, prepositions and so on.
# I'm also removing numbers

# %%
import nltk
from nltk.corpus import stopwords
nltk.download("stopwords")
nltk.download('italian')  # Downloading Italian stopwords

# %%
import re
italian_stopwords = stopwords.words("italian")
def remove_stopwords(sentence):
    """Removes stopwords from the Italian language (so mostly articles, conjunctions, etc)"""
    words = nltk.word_tokenize(sentence) 
    filtered_words = [word for word in words if word.lower() not in italian_stopwords]
    cleaned_sentence = ' '.join(filtered_words)  # Join the words back into a sentence
    return cleaned_sentence

def remove_dates_and_numbers(text):
    "removes dates and numbers. The regex could be better but in the end it did the trick, so i'm not adding complexity"
    words = nltk.word_tokenize(text)  # Tokenize the text into words
    
    # Filter out dates and numbers using regex
    filtered_words = [word for word in words if not re.match(r'^\d{1,2}-\d{1,2}-\d{1,4}$', word) and not word.isdigit()]
    
    return ' '.join(filtered_words)  # Join the filtered words back into a string

# Define the main function to remove stopwords and dates/numbers
def clean_text(text):
    cleaned_text = remove_stopwords(text)  # First, remove stopwords
    cleaned_text = remove_dates_and_numbers(cleaned_text)  # Then, remove dates and numbers
    return cleaned_text


# %%
df["tfidf_text"] = df["text"].map(clean_text)
problematic_df = df[df["problematic_content_label_present"]]

# %% [markdown]
# I'm using the "problematic_content_label_present" to filter for problematic content. It's not super correct, because there may be some annotator disagreement, but I'm also applying a "better safe then sorry logic". If a sample has been deemed problematic, I'll just discard it

# %%
sentences = df["tfidf_text"].tolist()
problematic_sentences = problematic_df["tfidf_text"].tolist()

# %% [markdown]
# ## TF-IDF and keyword extraction

# %%
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
vectorizer = vectorizer.fit(sentences)


# %%
def extract_top_k_keywords(vectorizer, sentence, k=10):
    """
    Extracts the top K keywords from a single sentence using TF-IDF scores.
    
    Parameters:
        vectorizer (TfidfVectorizer): The fitted TfidfVectorizer object.
        sentence (str): A single sentence or text string.
        k (int): Number of top keywords to extract. Default is 10.
        
    Returns:
        list of tuples: List of tuples containing the keyword and its TF-IDF score, sorted by descending order of scores.
    """
    tfidf_matrix = vectorizer.transform([sentence])
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = zip(feature_names, tfidf_matrix[0].toarray()[0])
    sorted_tfidf_scores = sorted(tfidf_scores, key=lambda x: x[1], reverse=True)
    return sorted_tfidf_scores[:k]



# %% [markdown]
# With a quick glance, I would say that most of the problematic content can be categorized in the following categories:
# - pornografy:
# - gambling
# - innocous ads

# %%
def print_best_keywords(sentences):
    for idx, p_sent in enumerate(sentences):
        keyword_score_pairs = extract_top_k_keywords(vectorizer, p_sent, k=n_keywords)
        print(f"Example {idx:02d}: {[keyword for keyword, score in keyword_score_pairs]}")


# %%
for idx, p_sent in enumerate(problematic_sentences):
    keyword_score_pairs = extract_top_k_keywords(vectorizer, p_sent, k=n_keywords)
    print(f"Example {idx:02d}: {[keyword for keyword, score in keyword_score_pairs]}")

# %% [markdown]
# Let's sample some random good quality samples and let's see roughly the distribution. I'm ignoring the one with problematic content.

# %%
classes = ["None", "Minimal", "Basic", "Good", "Excellent"]
value_mapping = {
    "None":0,
    "Minimal":1,
    "Basic":2,
    "Good":3,
    "Excellent":4,
    "❗ Problematic Content ❗":-1
}


# %%
def best_educational_value(values):
    """
    Given a list of educational values, returns the value with the highest mapped score.
    Uses value_mapping for comparison.
    """
    return max(values, key=lambda x: value_mapping.get(x, 0))



# %%

# %%
df['best_educational_value'] = df['educational_value_labels'].apply(best_educational_value)
safe_df = df[df["problematic_content_label_present"] == False]

# %% [markdown]
# ## Excellent
# Looking at the outputs, il looks like that there are some samples about human biology and historical events.

# %%
excellent_samples = safe_df[safe_df["best_educational_value"]=="Excellent"]
len(excellent_samples)

# %%
excellent_sents = df[df['educational_value_labels'].apply(lambda x: 'Excellent' in x)]["tfidf_text"].to_list()

# %% [markdown]
#

# %%
excellent_sents

# %%
print_best_keywords(excellent_sents)

# %%
filter_class = "Good"
samples = safe_df[safe_df["best_educational_value"]==filter_class]
print("Number of good samples", len(samples))
class_sentences = samples["tfidf_text"].to_list()

# %%
class_sentences[0:10]

# %%
print_best_keywords(class_sentences[0:11])

# %% [markdown]
# ## Basic

# %%
filter_class = "Basic"
samples = safe_df[safe_df["best_educational_value"]==filter_class]
print("Number of good samples", len(samples))
class_sentences = samples["tfidf_text"].to_list()

# %%
print_best_keywords(class_sentences[0:11])

# %%
df['tfidf_keywords'] = df["tfidf_text"].apply(
        lambda x: [keyword for keyword, score in extract_top_k_keywords(vectorizer, x, k=n_keywords)]
    )

# %%
df

# %%
df[df['educational_value_labels'].apply(lambda x: 'Excellent' in x)]["tfidf_text"].to_list()

# %%
df.to_parquet(product["dataset"])

# %%
