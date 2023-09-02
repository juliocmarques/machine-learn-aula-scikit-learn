import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Campo alvo de previsão
avaliar =  'score_credito' #'Cliente_Alvo'

# Conversão para númerico
def converter_colunas_numericas(df):
    lista_colunas_convertidas = []
    for coluna in df.columns:
        # Verifica se a coluna é do tipo 'object'
        if df[coluna].dtype == 'object':
            # Tenta converter os valores da coluna para float, ignorando erros
            try:
                # Verifique se todos os valores na coluna podem ser convertidos para float
                if df[coluna].apply(lambda x: pd.to_numeric(x, errors='coerce')).notna().all():
                    df[coluna] = pd.to_numeric(df[coluna], errors='coerce')
                    lista_colunas_convertidas.append(coluna)
                    print("Colunas convertidas: ", lista_colunas_convertidas)
            except ValueError:
                pass  # Não foi possível converter, mantenha como 'object'
    return df

# #tratamento de valores ausentes:
def tratar_valores_ausentes(df):
    lista_ausentes_drop = []
    lista_ausentes_mode = []
    lista_ausentes_mean = []

    for coluna in df.columns:

        limiar = 0.1 # Limite % de nulos considerando exclusão da base
        porcentagem_nulos = df[coluna].isnull().sum() / len(df) #* 100

        if coluna != avaliar and porcentagem_nulos > 0:
            if porcentagem_nulos < limiar: 
                # Exclua os registros nulos na coluna
                df = df.dropna(subset=[coluna])
                lista_ausentes_drop.append(coluna)
                print("Excluídos: ", lista_ausentes_drop)

            elif df[coluna].dtype == 'object':
                # Preencha valores ausentes com o valor mais frequente (modo)
                df[coluna].fillna(df[coluna].mode()[0], inplace=True)
                lista_ausentes_mode.append(coluna)
                print("Substituidos pela moda: ", lista_ausentes_mode)

            elif np.issubdtype(df[coluna].dtype, np.number):
                # Preencha valores ausentes com a média
                df[coluna].fillna(df[coluna].mean(), inplace=True)
                lista_ausentes_mean.append(coluna)
                print("Substituidos pela média: ", lista_ausentes_mean)
            else:
                print("Não há dados ausentes")                
    return df

# ou remava uma lista de colunas com alta cardinalidade
def remove_alta_cardinalidade(df):
    
    if len(df) > 99000: 
        limite_superior_volume = 0.10
    else:
        limite_superior_volume = 1.10

    limite_superior = len(df) * limite_superior_volume # Calcula o limite superior de contagem distintos (10% do total de registros)
    colunas_para_excluir = [] # Lista para armazenar as colunas a serem excluídas
    contagem_distinta = df.nunique() # Calcula a contagem distintos para cada coluna
    lista_percentual = []
    # Itera pelas colunas e verifica se a contagem é superior ao limite superior
    for coluna, contagem in contagem_distinta.items():
        if contagem > limite_superior and coluna != avaliar:
            colunas_para_excluir.append(coluna)
            per_coluna =  (contagem / len(df)) * 100
            lista_percentual.append(per_coluna)
            print("Colunas excluídas: ", colunas_para_excluir)
            print("Colunas excluídas %: ", lista_percentual)

    df_x = df.drop(colunas_para_excluir, axis=1) # Remove as colunas com contagem alta
    return colunas_para_excluir, df_x


# Ou converta todas as colunas object em númerico pois os modelos não trabalham com texto
def transforma_tipo_paramodelo(df):
    tranforme = LabelEncoder()
    lista_campos_transformados = []
    for col in df.columns:
        if df[col].dtype == "object" and col != avaliar:
            df[col] = tranforme.fit_transform(df[col])            
            lista_campos_transformados.append(col)
    print("Campos transformados: ", lista_campos_transformados)
    return lista_campos_transformados, df
