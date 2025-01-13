from flask import Flask, request, jsonify
import pickle
import pandas as pd
import os
from dotenv import load_dotenv
from flask_cors import CORS
from model import SRSModel

load_dotenv()



dataset_path = os.getenv("DATASET_PATH", "datasets/dataset.csv")
model_path = os.getenv("MODEL_PATH", "model/")
default_dataset_id = '2023_spotify_ds1.csv'
app_port = os.getenv("APP_PORT")
model = SRSModel()

print(f"Loading dataset from {dataset_path}...")
playlists_df  = pd.read_csv(os.path.join(dataset_path, default_dataset_id), low_memory=False)

print(f"Loading rules from {model_path}...")
if not os.path.exists(os.path.join(model_path, 'rules.pkl')):
     print(f"Rules not found\nCreating rules with default dataset: {model_path}{default_dataset_id}...")
     model.update_model(default_dataset_id)

app = Flask(__name__)
CORS(app)


@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    artist = data[0].get('artist_name')
    song = data[0].get('track_name')

    #Obtem os dados do modelo
    model_info = get_model_info()
    model_date = model_info[1]
    model_version = model_info[0]

    item_uri = playlists_df[(playlists_df['artist_name'] == artist) &
                            (playlists_df['track_name'] == song)]['track_uri']

    if item_uri.empty:
            return jsonify({'error': 'Song not found in the model.', 'model_date': model_date, 'model_version': model_version})

    item_uri = item_uri.iloc[0]
    recomendacoes = model.rules[model.rules['antecedents'].apply(lambda x: item_uri in x)]

    # Retorne as músicas recomendadas
    recommended_uris = [list(rule['consequents'])[0] for _, rule in recomendacoes.iterrows()]
    filtered_playlists = playlists_df[playlists_df['track_uri'].isin(recommended_uris)]
    filtered_playlists = filtered_playlists.drop_duplicates(subset='track_uri', keep='first')
    result = filtered_playlists[['artist_name', 'track_name']].to_dict(orient='records')

    return jsonify({'recommendations': result, 'model_date': model_date, 'model_version': model_version})

@app.route('/update-model', methods=['POST'])
def update_model():
     data = request.json
     dataset_id = data.get('dataset_id')
     if dataset_id == None:
          dataset_id = default_dataset_id
     
     model_data = model.update_model(dataset_id)
     rsp = {
          "message" : "Model Updated Successfuly",
          "model_version": model_data.get('model_version'),
          "model_date": model_data.get('model_date')
     }

     return jsonify(rsp)

@app.route('/tracks', methods=['GET'])
def get_tracks():
     try:
          csv_path = os.path.join(dataset_path, '2023_spotify_songs.csv')

          df = pd.read_csv(csv_path)

          songs = df.to_dict(orient='records')
          #Obtem os dados do modelo
          model_info = get_model_info()
          model_date = model_info[1]
          model_version = model_info[0]

          return jsonify({'songs': songs, 'model_date': model_date, 'model_version': model_version}), 200
     except FileNotFoundError:
          return jsonify({'error': f"CSV file not found"}), 404
     except Exception as e:
          return jsonify({'error': str(e)})


def get_model_info():
    model_info_path = os.path.join(model_path, "model_info")
    model_date = ''
    model_version = ''
    if os.path.exists(model_info_path):
     with open(model_info_path, "r") as f:
          model_info = str(f.read()).split(";")
          model_version = model_info[0]
          model_date = model_info[1]
     
     return model_version, model_date
     

if __name__ == '__main__':
    print("app_port: "+str(app_port))
    app.run(host='0.0.0.0', port=app_port)