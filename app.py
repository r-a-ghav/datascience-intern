from flask import Flask, render_template, request
import pandas as pd
import joblib
from sklearn.impute import SimpleImputer
app = Flask(__name__)

model = joblib.load('auto_mpg_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

        cylinders = float(request.form['cylinders'])
        displacement = float(request.form['displacement'])
        horsepower = float(request.form['horsepower'])
        acceleration = float(request.form['acceleration'])
        modelyear = float(request.form['modelyear'])
        origin = float(request.form['origin'])
        weight = float(request.form['weight'])

        input_data = {'Cylinders': [cylinders], 'Displacement': [displacement], 'Horsepower': [horsepower],'Weight' : [weight], 'Acceleration' : [acceleration] , 'Model Year' : [modelyear], 'Origin' : [origin]}

        input_df = pd.DataFrame(input_data)

        predictions = model.predict(input_df)

        result = f'Predicted MPG: {predictions[0]:.2f}'
        return render_template('index.html', result=result)


if __name__ == '__main__':
    app.run(port=5000)
