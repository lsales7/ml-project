import pandas as pd
from features.build_rfm import montar_rfm_com_target

#Carregando dataset
df = pd.read_csv('data/processed/olist_processado.csv')

data_corte = '2018-07-19'
data_maxima = '2018-10-17'

#Calcular rfm 
rfm_df = montar_rfm_com_target(df, data_corte, data_maxima)

print(rfm_df.head())
print(f"Linhas: {rfm_df.shape[0]}")
print(f"Colunas: {rfm_df.shape[1]}")
print(rfm_df.describe())
