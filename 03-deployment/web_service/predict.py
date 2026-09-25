import pickle

from flask import Flask, request,jsonify

with open('models/lin_reg.bin', 'rb') as f_in:
    (dv, model) = pickle.load(f_in)


def prepare_features(ride):
    features = {}
    features['PU_DO'] = '%s_%s' % (ride['PULocationID'], ride['DOLocationID'])
    features['trip_distance'] = ride['trip_distance']
    return features

def predict(features):
    X = dv.transform(features)
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