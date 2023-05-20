from flask import Flask, render_template, request, redirect, url_for
from src.pipelines.predict_pipeline import CustomData, PredictPipeline
from sklearn import set_config

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method=='GET':
        return render_template('index.html')
    else:
        # Variable to pass prediction to other functions
        global prediction

        # Initializing data
        data = CustomData(
            book = request.form.get('book'),
            delivery = request.form.get('delivery'),
            rate = request.form.get('rate'),
            votes = request.form.get('votes'),
            location = request.form.get('location'),
            type_tag = (', ').join(request.form.getlist('type_tag')),
            r_type= request.form.get('r_type')
            )
        
        # Setting sklearn global configurations
        set_config(transform_output="pandas")

        pred_df = data.get_data_as_dataframe()  

        prediction = PredictPipeline().predict(pred_df)

        return redirect(url_for('results'))

@app.route('/results')
def results():
    return render_template('result.html', prediction = prediction)

if __name__=="__main__":
    app.run(debug=True, port=5000)