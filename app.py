from flask import Flask, request, jsonify
import pickle
import pandas as pd
import os
from dotenv import load_dotenv
from flask_cors import CORS

load_dotenv()



dataset_path = os.getenv("DATASET_PATH", "datasets/dataset.csv")
app_port = os.getenv("APP_PORT")

print(f"Loading dataset from {dataset_path}.")
playlists_df  = pd.read_csv(dataset_path, low_memory=False)

app = Flask(__name__)
CORS(app)

with open('rules.pkl', 'rb') as f:
    rules = pickle.load(f)


@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    artist = data[0].get('artist_name')
    song = data[0].get('track_name')

    item_uri = playlists_df[(playlists_df['artist_name'] == artist) &
                            (playlists_df['track_name'] == song)]['track_uri']

    if item_uri.empty:
            return jsonify({'error': 'Song not found in the model.'})

    item_uri = item_uri.iloc[0]
    recomendacoes = rules[rules['antecedents'].apply(lambda x: item_uri in x)]

    # Retorne as músicas recomendadas
    recommended_uris = [list(rule['consequents'])[0] for _, rule in recomendacoes.iterrows()]
    filtered_playlists = playlists_df[playlists_df['track_uri'].isin(recommended_uris)]
    filtered_playlists = filtered_playlists.drop_duplicates(subset='track_uri', keep='first')
    result = filtered_playlists[['artist_name', 'track_name']].to_dict(orient='records')
    return jsonify({'recommendations': result})

if __name__ == '__main__':
    print("app_port: "+str(app_port))
    app.run(host='0.0.0.0', port=app_port)