import pandas as pd

def download_parquet(product, uri):
    df = pd.read_parquet(uri)
    df.to_parquet(product, index=False)