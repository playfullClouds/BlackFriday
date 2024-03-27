import numpy as np

from flask import Flask, request, render_template

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

@app.route('/')
def index():
    # Render the landing page
    return render_template("index.html")

@app.route('/predict', methods=['GET'])
def predict():
    # Render the prediction input page
    return render_template("home.html")

@app.route('/predictdata', methods=['POST'])
def predict_data():
    # Extract form data, perform prediction, and show results
    data = CustomData(
        Gender=request.form.get('Gender'),
        Age=request.form.get('Age'),
        Occupation=int(request.form.get('Occupation')),
        City_Category=request.form.get('City_Category'),
        Stay_In_Current_City_Years=int(request.form.get('Stay_In_Current_City_Years')),
        Marital_Status=int(request.form.get('Marital_Status')),
        Product_Category_1=int(request.form.get('Product_Category_1')),
        Product_Category_2=int(request.form.get('Product_Category_2') if request.form.get('Product_Category_2') else 0),
        Product_Category_3=int(request.form.get('Product_Category_3') if request.form.get('Product_Category_3') else 0)
    )

    pred_df = data.get_data_as_data_frame()
    predict_pipeline = PredictPipeline()
    results = predict_pipeline.predict(pred_df)
    
    
    # results = results[0] if isinstance(results, (list, tuple)) else results
    # formatted_results = "${:,.2f}".format(results)

    # return render_template("result.html", results=formatted_results)
    
    # Check if results is a numpy array and extract the first value if it is
    if isinstance(results, np.ndarray):
        result = results.item(0)  # Convert the first element to a Python scalar
    else:
        result = results  # Assume results is already a scalar if not an array
    
    # Format the result
    formatted_result = "{:,.2f}".format(result)

    # Pass the formatted result to your template
    return render_template("result.html", results=formatted_result)

if __name__ == '__main__':
    app.run(debug=True)
