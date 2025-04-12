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
# If this task has dependencies, declare them in the YAML spec and leave this
# as None
upstream = None

# This is a placeholder, leave it as None
product = None


# %% tags=["injected-parameters"]
# Parameters
embedder_model = "nickprock/sentence-bert-base-italian-uncased"
upstream = {"extract_topic_llama": "/home/oscar/Progetti/fineweb-c-analysis-ita/products/models/llama_topics.parquet", "extract_topic_gemma": "/home/oscar/Progetti/fineweb-c-analysis-ita/products/models/gemma_topics.parquet", "tf_idf_keywords": {"nb": "/home/oscar/Progetti/fineweb-c-analysis-ita/products/features/tf_idf_keywords.ipynb", "dataset": "/home/oscar/Progetti/fineweb-c-analysis-ita/products/features/df_with_keywords.parquet"}}
product = {"nb": "/home/oscar/Progetti/fineweb-c-analysis-ita/products/analyze/analyze_llama_topics.ipynb", "plotly_diagrams": "/home/oscar/Progetti/fineweb-c-analysis-ita/products/analyze/plotly_diagrams"}


# %%
import pandas as pd

# %%
import os
os.environ["TOKENIZERS_PARALLELISM"] = "0"

# %%
gemma_df = pd.read_parquet(upstream["extract_topic_gemma"])
llama_df = pd.read_parquet(upstream["extract_topic_llama"])

dataset = pd.read_parquet(upstream["tf_idf_keywords"]["dataset"])

# %% [markdown]
# Let's see how many and which categories have been identified by gemmaa and llama

# %%
import pandas as pd

# Assuming gemma_df and llama_df are already defined based on your context
# You need to merge these dataframes with dataset on the "id" column
dataset = pd.merge(dataset, gemma_df, on='id', how='left')
dataset.rename(columns={"category":"category_gemma"}, inplace=True)


# %%
dataset = pd.merge(dataset, llama_df, on='id', how='left')
dataset = dataset.rename(columns={"category":"category_llama"})

# %%
gemma_categories = dataset["category_gemma"].unique().tolist()
llama_categories = dataset["category_llama"].unique().tolist()
print("Number of gemma categories:", len(gemma_categories))
print("Number of llama categories:", len(llama_categories))

# %%
from typing import List


def clean_gemma_categories(categories: List[str]):
    categories = [c for c in categories if "CATEGORIA" not in c]
    categories = [c for c in categories if "=" not in c]
    categories = [c for c in categories if "d-none" not in c.lower()]
    categories = [c for c in categories if c != ""]
    return categories
    


# %%
print(len(gemma_categories))
gemma_categories = clean_gemma_categories(gemma_categories)
print(len(gemma_categories))

# %%
dataset["text"]

# %%
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch
from tqdm import tqdm

# Load the tokenizer and model from Hugging Face
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"
print("Using device:",device)
model = SentenceTransformer(embedder_model, device=device)

# Example list of texts (you can replace this with your actual data)
texts = dataset["text"].tolist()


# %%
# Get the embeddings for the list of texts
embeddings = model.encode(texts, show_progress_bar=True)

# %%
categories_embeddings = model.encode(dataset["category_llama"].tolist())

# %%
len(dataset["category_llama"].unique().tolist()), len(dataset["category_gemma"].unique().tolist())

# %%
# dataset["category_llama"].hist(xrot=45, xlabelsize=10)

# %%
alpha = 0.2
overal_embs = alpha *embeddings + (1-alpha) *categories_embeddings 

# %%

# %%
# Apply PCA for dimensionality reduction
pca = PCA(n_components=2)
print("Applying PCA")
reduced_data_pca = pca.fit_transform(overal_embs)
# # Apply t-SNE for dimensionality reduction
# print("Applying T-TSNE")
# tsne = TSNE(n_components=2, perplexity=30, learning_rate='auto', random_state=42)
# reduced_data_tsne = tsne.fit_transform(overal_embs)




# %%
dataset["best_educational_value"].unique()

# %%
categories = dataset["category_llama"].unique()

# %%
dataset["x"] = [ar[0] for ar in reduced_data_pca]
dataset["y"] = [ar[1] for ar in reduced_data_pca]

# %%
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

symbol_map = {
    'None': 'circle',
    'Minimal': 'square',
    'Basic': 'diamond',
    'Good': 'cross',
    'Excellent': 'star',
    '❗ Problematic Content ❗': 'x'
}

# Helper function to truncate text to 50 words and add line breaks every few words
def truncate_text(text, max_words=50, words_per_line=8):
    words = text.split()
    if len(words) > max_words:
        words = words[:max_words]
        words.append('...')
    
    # Add line breaks every words_per_line words
    result = []
    for i in range(0, len(words), words_per_line):
        result.append(' '.join(words[i:i+words_per_line]))
    
    return '<br>'.join(result)

# Create a custom hover template with text wrapping
hover_template = (
    "<b>Text:</b> %{customdata[3]}<br>" +
    "<b>Category:</b> %{customdata[1]}<br>" +
    "<b>Educational Value:</b> %{customdata[2]}<extra></extra>"
)

# Create figure using Plotly Graph Objects for more control
fig = go.Figure()

# Create a separate trace for each educational value to control the legend
for edu_value in dataset["best_educational_value"].unique():
    # Filter dataset for current educational value
    df_filtered = dataset[dataset['best_educational_value'] == edu_value]
    
    if not df_filtered.empty:
        # Truncate text and add line breaks for display in hover
        formatted_texts = [truncate_text(text, 50, 8) for text in df_filtered['text']]
        
        # Add trace for this educational value
        fig.add_trace(go.Scatter(
            x=df_filtered['x'],
            y=df_filtered['y'],
            mode='markers',
            marker=dict(
                symbol=symbol_map[edu_value],
                color=df_filtered['category_llama'].map(
                    {cat: px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)] 
                     for i, cat in enumerate(categories)}
                ),
                size=10,
                line=dict(color='DarkSlateGrey')
            ),
            customdata=np.stack((
                df_filtered['text'],  # Keep original text (not used in hover)
                df_filtered['category_llama'], 
                df_filtered['best_educational_value'],
                formatted_texts  # Add formatted text with line breaks for hover display
            ), axis=-1),
            hovertemplate=hover_template,
            name=edu_value,
            legendgroup=edu_value,
            showlegend=True
        ))

# Customize the layout
fig.update_layout(
    title='Text Samples Visualization',
    xaxis=dict(
        title='X Dimension',
        showgrid=True,
        gridwidth=0.5,
        gridcolor='lightgray',
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='black'
    ),
    yaxis=dict(
        title='Y Dimension',
        showgrid=True,
        gridwidth=0.5,
        gridcolor='lightgray',
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='black'
    ),
    legend=dict(
        title='Educational Value',
        orientation="h",
        yanchor="bottom",
        y=-0.2,
        xanchor="center",
        x=0.5
    ),
    plot_bgcolor='white',
    margin=dict(l=20, r=20, t=60, b=60),
    autosize=True,
    height=600,
    hoverlabel=dict(
        bgcolor="white",
        font_size=12,
        font_family="Arial",
        align="left"
    )
)

# Display the figure
fig.show()

# %%
from pathlib import Path


Path(product["plotly_diagrams"]).mkdir(parents=True, exist_ok=True)
fig.write_html(Path(product["plotly_diagrams"]) / "sample_scatter.html")



# %%
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Create a count of categories
category_counts = dataset['category_llama'].value_counts().reset_index()
category_counts.columns = ['category', 'count']

# Sort by count in descending order and take top 20
top_k = 23
category_counts = category_counts.sort_values('count', ascending=False).head(top_k)
# Re-sort for display (ascending so largest bars are at the top of the plot)
category_counts = category_counts.sort_values('count', ascending=True)

# Create horizontal bar chart
fig_hist = go.Figure()

fig_hist.add_trace(go.Bar(
    y=category_counts['category'],
    x=category_counts['count'],
    orientation='h',
    marker=dict(
        color=[px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)] 
               for i, _ in enumerate(category_counts['category'])],
        line=dict(color='DarkSlateGrey', width=1)
    ),
    text=category_counts['count'],
    textposition='auto',
))

# Update layout
fig_hist.update_layout(
    title=f'Top {top_k} Categories by Count',
    xaxis_title='Count',
    yaxis_title='Category',
    plot_bgcolor='white',
    margin=dict(l=20, r=20, t=60, b=20),
    xaxis=dict(
        showgrid=True,
        gridwidth=0.5,
        gridcolor='lightgray',
    ),
    yaxis=dict(
        showgrid=False,
    ),
    # Make the plot larger and adjust height based on number of categories
    height=600,  # Increased height to accommodate 20 categories
    width=800,   # Wider plot
)

# Display the histogram
fig_hist.show()

# Save the histogram
fig_hist.write_html(Path(product["plotly_diagrams"]) / "class_distribution_llama.html")

# %%
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Create a count of categories
category_counts = dataset['category_gemma'].value_counts().reset_index()
category_counts.columns = ['category', 'count']

# Sort by count in descending order and take top 20
top_k = 23
category_counts = category_counts.sort_values('count', ascending=False).head(top_k)
# Re-sort for display (ascending so largest bars are at the top of the plot)
category_counts = category_counts.sort_values('count', ascending=True)

# Create horizontal bar chart
fig_hist = go.Figure()

fig_hist.add_trace(go.Bar(
    y=category_counts['category'],
    x=category_counts['count'],
    orientation='h',
    marker=dict(
        color=[px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)] 
               for i, _ in enumerate(category_counts['category'])],
        line=dict(color='DarkSlateGrey', width=1)
    ),
    text=category_counts['count'],
    textposition='auto',
))

# Update layout
fig_hist.update_layout(
    title=f'Top {top_k} Categories by Count',
    xaxis_title='Count',
    yaxis_title='Category',
    plot_bgcolor='white',
    margin=dict(l=20, r=20, t=60, b=20),
    xaxis=dict(
        showgrid=True,
        gridwidth=0.5,
        gridcolor='lightgray',
    ),
    yaxis=dict(
        showgrid=False,
    ),
    # Make the plot larger and adjust height based on number of categories
    height=600,  # Increased height to accommodate 20 categories
    width=800,   # Wider plot
)

# Display the histogram
fig_hist.show()

# Save the histogram
fig_hist.write_html(Path(product["plotly_diagrams"]) / "class_distribution_gemma.html")


# %%

def plotly_plot(data, category_column='category', class_column='class'):
    # Set a fixed width and height for the figure to be more square-shaped
    fixed_size = 800  # You can adjust this value to make the plot wider or taller
    
    fig = px.scatter(data_frame=data, x='x', y='y', color=category_column, symbol=class_column, title='Visualization of Embeddings', width=fixed_size, height=fixed_size)
    
    # Adding hover information to show both category and class
    text_hover = [f'Text: {data['text']}<br>Class: {data[category_column].iloc[i]}<br>Category: {data[class_column].iloc[i]}' for i in range(len(data))]
    fig.update_traces(hoverinfo='text', text=text_hover)
    # fig.update_layout(showlegend=False)
    
    return fig


# %%
dataset.columns

# %%

# %%
fig =plotly_plot(dataset,category_column="category_llama", class_column="best_educational_value")

# %%
fig.show()

# %%
plotly_html = fig.to_html(full_html=False, include_plotlyjs='cdn')

# %%
from pathlib import Path
# %%
embeddings

# %%
# # Apply t-SNE for dimensionality reduction
# print("Applying T-TSNE")
# tsne = TSNE(n_components=2, perplexity=30, learning_rate='auto', random_state=42)
# reduced_data_tsne = tsne.fit_transform(embeddings)


# %%

# Print the reduced data
print("PCA Reduced Data:")
print(reduced_data_pca)
