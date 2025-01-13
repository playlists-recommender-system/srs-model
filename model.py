import os
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules
import pickle
from datetime import datetime

class SRSModel:
    def __init__(self):
        self.dataset_folder = os.getenv("DATASET_PATH", "./datasets")
        self.model_folder = os.getenv("MODEL_PATH", './models')
        self.rules = None
        os.makedirs(self.model_folder, exist_ok=True)
    
    def _load_dataset(self, dataset_id):
        dataset_path = os.path.join(self.dataset_folder, f"{dataset_id}")
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset Not Found: {dataset_path}")
        return pd.read_csv(dataset_path, low_memory=False)
    
    def _preprocess_data(self, playlists):
        transactions = playlists.groupby('pid')['track_uri'].apply(list).tolist()
        transactions = [list(set(transaction)) for transaction in transactions]
        return transactions
    
    def _encode_transactions(self, transactions):
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        return pd.DataFrame(te_ary, columns=te.columns_)
    
    def _generate_rules(self, df):
        frequent_itemsets = fpgrowth(df, min_support=0.05, use_colnames=True)
        rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1.0, num_itemsets=None)
        filtered_rules = rules[(rules['antecedents'].apply(len) == 1) & (rules['consequents'].apply(len) == 1)]
        return filtered_rules
    
    def _save_rules(self, rules):
        model_date = str(datetime.now())
        model_version = 1
        
        model_path = os.path.join(self.model_folder, f"rules.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(rules, f)
        
        if os.path.exists(os.path.join(self.model_folder, "model_info")):       
            with open(os.path.join(self.model_folder, "model_info"), "r") as f:
                _info = f.read()
                info = str(_info).split(";")
                model_version = int(info[0]) + 1
        with open(os.path.join(self.model_folder, "model_info"), "w") as f:
            f.write(f"{model_version};{model_date}")

        with open(model_path, 'rb') as f:
            self.rules = pickle.load(f)
        
        return {"model_version": model_version, "model_date": model_date}

    
    def update_model(self, dataset_id):
        try:
            print(f"Loading dataset {dataset_id}...")
            playlists = self._load_dataset(dataset_id)

            print(f"Preprocessing data...")
            transactions = self._preprocess_data(playlists)

            print(f"Encoding transactions...")
            df = self._encode_transactions(transactions)

            print(f"Generating association rules...")
            rules = self._generate_rules(df)

            print(f"Saving association rules...")
            model_data = self._save_rules(rules)

            print(f"Model successfully updated!")
            return model_data
        except Exception as e:
            print(f"Faled to update model: {e}")

if __name__ == "__main__":
    updater = ModelUpdater()
    dataset_id = "2023_spotify_ds1"
    updater.update_model(dataset_id)
