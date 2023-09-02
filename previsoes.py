from modelos_treinamento import *

# Criando uma nova base de dados fictícia para fazer previsões
# nova_base = {
#     'Idade': [28, 38, 45, 22, 59],
#     'Salario': [48000, 42000, 55000, 32000, 70000],
#     'Compras_Mensais': [2, 3, 1, 4, 6],
#     'Gênero': ['M', 'F', 'M', 'F', 'M'],
#     'Estado_Civil': ['Casado', 'Solteiro', 'Casado', 'Casado', 'Solteiro']
# }

df_nova_base = pd.read_csv('novos_clientes.csv')
# Criar um DataFrame Pandas com os novos dados
# df_nova_base = pd.DataFrame(nova_base)

df_sem_colunas_altas = df_nova_base.drop(colunas_para_excluir, axis=1) # Remove as colunas com contagem alta

tipo_transformados, df_previsao = transforma_tipo_paramodelo(df_sem_colunas_altas)

# Faça previsões usando os modelos treinados
# previsoes_logistic = logistic_model.predict(df_previsao)
previsoes_decision_tree = decision_tree_model.predict(df_previsao)
previsoes_random_forest = random_forest_model.predict(df_previsao)

# Exiba as previsões para cada modelo
# print("Previsões usando Regressão Logística:")
# print(previsoes_logistic)

print("\nPrevisões usando Árvore de Decisão:")
print(previsoes_decision_tree)

print("\nPrevisões usando Random Forest:")
print(previsoes_random_forest)
