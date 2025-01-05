import pandas as pd

playlists = pd.read_csv("../datasets/2023_spotify_ds1.csv",low_memory=False)

transactions = playlists.groupby('pid')['track_uri'].apply(list).tolist()

transactions = [list(set(transaction)) for transaction in transactions]

##################################################################
from mlxtend.preprocessing import TransactionEncoder

te = TransactionEncoder()

te_ary = te.fit(transactions).transform(transactions)

df = pd.DataFrame(te_ary, columns=te.columns_)

print(df.memory_usage(deep=True).sum() / (1024**2), "MB")


#################################################################

from mlxtend.frequent_patterns import apriori, association_rules

frequent_itemsets = apriori(df, min_support=0.05, use_colnames=True)
print(frequent_itemsets)
rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1.0, num_itemsets=None)

filtered_rules = rules[(rules['antecedents'].apply(len) == 1) & (rules['consequents'].apply(len) == 1)]


###############################################################

import pickle

with open('rules.pkl', 'wb') as f:
    pickle.dump(filtered_rules, f)

