import datetime
from flask import Flask, request, render_template
from flask_cors import CORS, cross_origin
from Pipeline_prediction.predictFromModel import Prediction
from Pipeline_training.trainingModel import train_model
from app_tracking.exception import AppException
from app_tracking.logger import App_Logger


application = Flask(__name__)
app = application
CORS(app)


@app.route("/", methods=['POST','GET'])
@cross_origin()
def home():
    return render_template('index.html')

@app.route("/about", methods=['POST','GET'])
@cross_origin()
def about():
    return render_template('index1.html')

@app.route("/trains", methods=['POST','GET'])
@cross_origin()
def trains():
    return render_template('index3.html')

@app.route("/train", methods=['POST','GET'])
@cross_origin()
def train():
    if request.method == 'POST':
        source = request.form.get('Local') or request.form.get('Cloud')
        if source:
            try:
                message = train_model(source)
                return render_template('index4.html', message=message)      
            except:
                message = "Oops! Something Went Wrong.. \n Contact me"
                return render_template('index4.html', message=message)
        else: 
            raise Exception("Option not selected")
    else:          
        message = "Something went wrong contact us  .."
        return render_template('index4.html', message=message)

@app.route("/predicts", methods=['POST','GET'])
@cross_origin()
def predicts():
    return render_template('index2.html')

@app.route("/predict", methods=['GET', 'POST'])
@cross_origin()
def predict():
    if request.method == 'POST':
        try:
            age = float(request.form['Age'])
            workclass = request.form['Workclass']
            fnlwgt = float(request.form['FNLWGT'])
            education = request.form['Education']
            marital_status = request.form['Marital Status']
            occupation = request.form['Occupation']
            relationship = request.form['Relationship']
            race = request.form['Race']
            sex = float(request.form['Sex'])
            capital_gain = float(request.form['Capital Gain'])
            capital_loss = float(request.form['Capital Loss'])
            hours_per_week = float(request.form['Hours/Week'])
            country = request.form['Country']

            data = [age, workclass, fnlwgt, education, marital_status,
                    occupation, relationship, race, sex, capital_gain,
                    capital_loss, hours_per_week, country]

            predict_object = Prediction(data=data)
            result = predict_object.prediction()
            return render_template("index5.html", result=result)
        except Exception as e:
            return render_template("index5.html", result="Oops! Something Went Wrong.. Please train the model first else \n Contact me")
            raise AppException(e,sys) from e
    else:  
        return render_template("index5.html", result="Something went wrong contact us  ..")
    

if __name__ == "__main__":
    app.run(host="0.0.0.0",port=5000)
