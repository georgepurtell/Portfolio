from flask import Flask, make_response, request, render_template
import io
from io import StringIO
import csv
import pandas as pd
import numpy as np
import pickle

# Initializing Flask
app = Flask(__name__)

@app.route('/', methods=["POST", "GET"])
def form():
    return render_template('home_page.html')

def transform(text_file_contents):
    return text_file_contents.replace("=", ",")

@app.route('/transform', methods=["POST", "GET"])
def transform_view():
    f = request.files['data_file']
    if not f:
        return "No file"

    stream = io.StringIO(f.stream.read().decode("UTF8"), newline=None)
    csv_input = csv.reader(stream)
    print(csv_input)
    for row in csv_input:
        print(row)

    stream.seek(0)
    result = transform(stream.read())

    df = pd.read_csv(StringIO(result))  
    
    # Cleaning data
    df.replace('$','', regex=True, inplace=True)

    df.replace(',','', regex=True, inplace=True)

    # Removing all catergorical variables
    df = df.select_dtypes('number')
    
    df.drop(columns=df.columns[0], axis=1, inplace=True)
    
    # Loading Linear Regression model
    loaded_model = pickle.load(open('model.pkl', 'rb'))  
    
    df['predicted_value'] = loaded_model.predict(df)

    response = make_response(df.to_csv())
    response.headers["Content-Disposition"] = "attachment; filename=predicted_result.csv"
    
    return response
    
if __name__ == '__main__':
    app.run(debug=True)
