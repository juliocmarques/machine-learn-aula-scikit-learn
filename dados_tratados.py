# Importe as bibliotecas necessárias
from funcoes_tratamento import *


# Crie uma tabela de dados fictícia com problemas comuns
# data = {
#     'Idade': [25, 30, None, 40, 45, 50, 55, 60, 65, 70],
#     'Salario': [30000, '35000', 40000, '45000', 50000, 55000, 60000, '65000', 70000, 75000],
#     'Compras_Mensais': [2, 3, 1, 3, 2, 4, 5, 4, 6, 5],
#     'Gênero': ['M', 'F', 'M', 'F', 'M', 'F', 'M', 'F', 'M', 'F'],
#     'Estado_Civil': ['Casado', 'Solteiro', 'Casado', 'Casado', 'Solteiro', 'Casado', 'Solteiro', 'Divorciado', 'Casado', 'Casado'],
#     'Cliente_Alvo': ['Sim', 'Não', 'Sim', 'Não', 'Sim','Sim','Sim','Não','Não','Não']    
# }


# Crie um DataFrame do Pandas
# df = pd.DataFrame(data)
df = pd.read_csv('clientes.csv')
print("-----------------------------importação dos dados efetuada-------------------------------------------------")

# Aplica a função para converter colunas dinamicamente
df_identifica_numericos = converter_colunas_numericas(df)


print("-----------------------------Identificação e Conversão de tipo efetuada------------------------------------")

# Aplica a função para tratamento de valores ausentes            
df_trata_ausentes = tratar_valores_ausentes(df_identifica_numericos)
# Lide com valores ausentes (substitua por média da idade e salário)
# df['Idade'].fillna(df['Idade'].mean(), inplace=True)
# df['Salario'] = pd.to_numeric(df['Salario'], errors='coerce')
# df['Salario'].fillna(df['Salario'].mean(), inplace=True)
# print(df.head())
# print(df.info())

print("-----------------------------Tratamento de dados ausentes efetuada-----------------------------------------")

# Remova colunas redundantes ou não informativas
# df.drop(columns=['Estado_Civil'], inplace=True)

# Aplica função para excluir colunas com alta cardinalidade, campos com id
colunas_para_excluir, df_trata_cardinalidade = remove_alta_cardinalidade(df_trata_ausentes)

print("-----------------------------Exclusão de campos com alta cardinalidade efetuada----------------------------")

# Converta a coluna "Gênero" em valores numéricos (codificação one-hot)
# df = pd.get_dummies(df, columns=['Gênero'], drop_first=True)

# Aplica função para converter object para numerico pois os modelos não trabalham com este tipo,
# sendo isso necessário para criar uma categorizacao numerica para cada tipo de object
tipo_transformados, df_converte_oject_paranumerico = transforma_tipo_paramodelo(df_trata_cardinalidade)

print("-----------------------------Conversão de tipos object para o modelo efetuada------------------------------")


# Remove dados duplicados e agrupa pela média
df_distinto = df_converte_oject_paranumerico.drop_duplicates()



# grupos = df_distinto.groupby([avaliar])
# media_por_campo = grupos.mean()

# # # # Visualize os dados após o tratamento
# # # # print("\nDados Após o Tratamento:")
# print(media_por_campo.info())
# print(media_por_campo.head())
