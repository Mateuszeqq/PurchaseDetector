import torch
from flask import Flask
from flask import request
from flask import jsonify
import json
import pickle
from models.SessionClassifier import SessionClassifier
from models.NaiveClassifier import NaiveClassifier


app = Flask(__name__)


def load_models(naive_path, nn_path):
    with open(naive_path, 'rb') as f:
        naive_model = pickle.load(f)

    nn_model = SessionClassifier(30, 7)
    nn_model.load_state_dict(torch.load(nn_path))
    nn_model.eval()

    return naive_model, nn_model


def log(r):
    with open('logs.txt', 'a') as f:
        json.dump(r, f)
        f.write('\n')


def get_nn_model_certainty(prediction, threshold):
    certainty = float(prediction)
    if int(prediction > threshold) == 0:
        certainty = round((1 - certainty) * 100, 2)
    else:
        certainty = round(certainty * 100, 2)
    return certainty


@app.route('/predict', methods=['POST'])
def predict():
    THRESHOLD = 0.5
    naive_model, nn_model = load_models('naive.pkl', 'model.pth')

    if request.method == 'POST':
        r = request.json
        if isinstance(r, str):
            r = json.loads(r)
        try:
            x_cat = r['x_cat']
            x_num = r['x_num']
            model = r['model']
        except Exception as e:
            print('WRONG REQUEST')
            return "Record not found", 400

        x_cat = torch.Tensor(x_cat)
        x_num = torch.Tensor(x_num)

        if model not in ['naive_model', 'nn_model']:
            return f"Model {model} doesn't exist", 400

        nn_prediction = nn_model(x_num, x_cat)
        nn_certainty = get_nn_model_certainty(nn_prediction, THRESHOLD)

        nn_prediction = int(nn_prediction > THRESHOLD)
        naive_prediction = int(naive_model.forward(x_num))

        r['naive_model'] = naive_prediction
        r['nn_model'] = nn_prediction
        r['nn_model_certainty'] = nn_certainty

        log(r)
        if model == 'naive_model':
            return jsonify({
                'result': naive_prediction
            })
        else:
            return jsonify({
                'certainty': nn_certainty,
                'result': nn_prediction
            })


if __name__ == '__main__':
    app.run(debug=True)
