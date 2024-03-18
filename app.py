from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load("fish_model.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    weight = float(request.form['weight'])
    length = float(request.form['length'])
    height = float(request.form['height'])
    width = float(request.form['width'])
    prediction = model.predict([[weight, length, height, width]])
    return render_template('index.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
