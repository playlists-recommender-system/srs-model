from flask import Flask, request, jsonify
import pickle
import pandas as pd
import os
from dotenv import load_dotenv
from flask_cors import CORS
from model_updater import ModelUpdater

load_dotenv()



dataset_path = os.getenv("DATASET_PATH", "datasets/dataset.csv")
model_path = os.getenv("MODEL_PATH", "model/")
default_dataset_id = '2023_spotify_ds1.csv'
app_port = os.getenv("APP_PORT")

print(f"Loading dataset from {dataset_path}.")
playlists_df  = pd.read_csv(os.path.join(dataset_path, default_dataset_id), low_memory=False)

print(f"Loading rules from {model_path}rules.pkl.")
if not os.path.exists(os.path.join(model_path, 'rules.pkl')):
     print(f"Creating rules with default dataset: {model_path} {default_dataset_id}.")
     updater = ModelUpdater()
     updater.update_model(default_dataset_id)

print(f"Loading rules from {model_path}rules.pkl.")
with open(os.path.join(model_path, 'rules.pkl'), 'rb') as f:
    rules = pickle.load(f)

app = Flask(__name__)
CORS(app)


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

@app.route('/update-model', methods=['POST'])
def update_model():
     data = request.json
     dataset_id = data.get('dataset_id')
     if dataset_id == None:
          return jsonify({'message': 'Must be sent a dataset'})
     
     updater = ModelUpdater()
     updater.update_model(dataset_id)
     return jsonify({'message' : "Model Updated Successfuly"})

@app.route('/tracks', methods=['GET'])
def get_tracks():
     try:
          csv_path = os.path.join(dataset_path, '2023_spotify_songs.csv')

          df = pd.read_csv(csv_path)

          records = df.to_dict(orient='records')

          return jsonify(records), 200
     except FileNotFoundError:
          return jsonify({'error': f"CSV file not found"}), 404
     except Exception as e:
          return jsonify({'error': str(e)})


if __name__ == '__main__':
    print("app_port: "+str(app_port))
    app.run(host='0.0.0.0', port=app_port)