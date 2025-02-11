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


# %%
import pandas as pd

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
llama_categories
