from flask import Flask, request, render_template

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

@app.route('/')
def index():
    # Render the landing page
    return render_template("index.html")

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_data():
    results = None  # Initialize results to None

    if request.method == 'POST':
        # Process form data and perform prediction
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
        print(pred_df)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        results = results[0]  # Get the first result for display

    # Render the same template whether it is a GET or POST request
    # Pass the results to the template, which will be None for a GET request
    return render_template('home.html', results=results)

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)
