from flask import Flask, render_template, request, redirect
import pandas as pd
import numpy as np
import pickle
import sklearn
import os

app = Flask(__name__)

IMG_FOLDER = os.path.join('static', 'img')
app.config['UPLOAD_FOLDER'] = IMG_FOLDER

cars = pd.read_csv('cleaned_quikr_car.csv')
model = pickle.load((open('LR_Model.pkl', 'rb')))


@app.route('/')
def index():
    # 'name', 'company', 'year', 'Price', 'kms_driven', 'fuel_type'
    cBrands = sorted(cars['company'].unique())
    cModels = sorted(cars['name'].unique())
    cYears = sorted(cars['year'].unique(), reverse=False)
    cFuel = sorted(cars['fuel_type'].unique(), reverse=False)
    return render_template('index.html', cBrands=cBrands, cModels=cModels, cYears=cYears, cFuel=cFuel)


@app.route('/predict', methods=['POST'])
def predict():
    car_Brand = request.form.get('cBrands')
    car_Model = request.form.get('cModels')
    car_Year = request.form.get('cYears')
    car_Fuel = request.form.get('cFuel')
    car_Km = request.form.get('cKm')

    print(car_Brand, car_Model, car_Year, car_Fuel, car_Km)

    prediction = model.predict(pd.DataFrame(columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'],
                                            data=np.array([car_Model,
                                                           car_Brand,
                                                           car_Year,
                                                           car_Km,
                                                           car_Fuel]).reshape(1, 5)))
    print(prediction)

    return str(np.round(prediction[0], 2))


if __name__ == "__main__":
    # app.run(debug=True)
    app.run(host='0.0.0.0', port=8080)
