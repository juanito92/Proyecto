import pandas as pd

def clean_data(df):
    # Limpiar datos (valores nulos, duplicados, etc.)
    df = df.dropna()
    return df
