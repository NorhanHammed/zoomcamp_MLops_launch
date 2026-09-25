import pickle

from flask import Flask, request,jsonify
from mlflow import pyfunc
import xgboost as xgb
from mlflow.tracking import MlflowClient
import mlflow

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
mlflow.set_experiment("nyc-taxi-experiment")
client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
run_id = "cc9d92685fae4814a8642ec621e8a965"

path = client.download_artifacts(run_id, "preprocessor/preprocessor.b")

with open(path, "rb") as f_in:
    dv = pickle.load(f_in)

logged_model = f"runs:/{run_id}/models_mlflow"
model = pyfunc.load_model(logged_model)

def prepare_features(ride):
    features = {}
    features['PU_DO'] = '%s_%s' % (ride['PULocationID'], ride['DOLocationID'])
    features['trip_distance'] = ride['trip_distance']

    return features

def predict(features):
    X = dv.transform(features)
    X = xgb.DMatrix(X)
    preds = model.predict(X)
    return float(preds[0])




app = Flask('trip-duration-prediction')


@app.route('/predict', methods=['POST'])
def predict_upon_request():
    ride_data = request.get_json()
    
    feature = prepare_features(ride_data)
    y = predict(feature)

    return jsonify({"trip_duration" : y}) 

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)