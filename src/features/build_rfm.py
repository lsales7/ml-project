import pandas as pd

# Data de recencia
def calcular_recencia(df, data_referencia):
    df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
    recencia = df.groupby('customer_unique_id')['order_purchase_timestamp'].max()
    recencia = (data_referencia - recencia).dt.days
    return recencia

#Frquencia de compra
def calcular_frequencia(df):
    frequencia = df.groupby('customer_unique_id')['order_id'].nunique()
    return frequencia

# Valor pago
def calcular_monetario(df):
    df['full_price'] = df['price'] + df['freight_value']
    monetario = df.groupby('customer_unique_id')['full_price'].sum()
    return monetario

# Tabela rfm com features
def montar_tabela_rfm(df, data_max):
    recencia = calcular_recencia(df, data_max)
    frequencia = calcular_frequencia(df)
    monetario = calcular_monetario(df)

    tabela = pd.concat([recencia, frequencia, monetario], axis=1)
    tabela.columns = ['recencia', 'frequencia', 'monetario']
    return tabela

#seprando data de treino (D) max - 90 dias
def filtrar_orders_treino(df, data_corte):
    df = df.copy()
    df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
    treino = df[(df['order_status'] == 'delivered') & (df['order_purchase_timestamp'] <= data_corte)]
    return treino

#target D+
def calcular_target(df, clientes, data_inicio, data_fim):
    df = df.copy()
    df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])

    janela = df[
        (df['order_status'] == 'delivered') &
        (df['order_purchase_timestamp'] > data_inicio) &
        (df['order_purchase_timestamp'] <= data_fim)
    ]

    compradores = janela.groupby('customer_unique_id')['order_id'].nunique()
    target = (compradores > 0).astype(int)
    target = target.reindex(clientes.index, fill_value=0)
    target.name = 'target'
    return target

#dataframe com a data do target
def montar_rfm_com_target(df, data_corte, data_fim_target):
    data_corte = pd.to_datetime(data_corte)
    data_fim_target = pd.to_datetime(data_fim_target)

    orders_treino = filtrar_orders_treino(df, data_corte)
    tabela_rfm = montar_tabela_rfm(orders_treino, data_corte)
    target = calcular_target(df, tabela_rfm, data_corte, data_fim_target)

    rfm_com_target = pd.concat([tabela_rfm, target], axis=1)
    rfm_com_target = rfm_com_target.reset_index()

    return rfm_com_target



