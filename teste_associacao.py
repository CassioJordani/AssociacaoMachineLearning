# bibliotecas
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Cada linha indica a transação do cliente e a coluna o produto. 
# Valor 1 foi comprado, 0 não comprado.
data = {
    'Notebook': [1, 1, 1, 1, 1],
    'Mouse': [1, 1, 0, 0, 1],
    'Teclado': [0, 0, 0, 0, 1],
    'Monitor': [1, 1, 1, 1, 1],
    'Cadeira Gamer': [0, 1, 1, 1, 1]
}

# Criando o dataframe
df = pd.DataFrame(data)

print("Transações de compras:")
print(df)

# a combinação deve ter pelo menos 80% das transações.
frequent_itemsets = apriori(df, min_support=0.8, use_colnames=True)

# mostrar itens frequentes
print("\nConjuntos de itens frequentes:")
print(frequent_itemsets)

# Gerando regras de associação a partir dos itens frequentes
# A métrica 'confidence' é usada para avaliar a relevância das regras.
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.7)

# Exibindo as regras de associação
print("\nRegras de associação:")
print(rules[['antecedents', 'consequents', 'support', 'confidence']])

# Imprimindo recomendações com base nas regras geradas
print("\nRecomendações de produtos:")
for index, rule in rules.iterrows():
    print(f"Se o cliente comprou {list(rule['antecedents'])}, recomendamos: {list(rule['consequents'])}")