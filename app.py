from flask import Flask, render_template, request
import pickle
import pandas as pd
# Initialise the Flask app
app = Flask("Flaskapp")
# Use pickle to load in the pre-trained model
filename = "models/model.sav"
model = pickle.load(open(filename, "rb"))
# Set up the main route
@app.route('/', methods=["GET", "POST"])
def main():
    if request.method == "POST":
        # Extract the input from the form
        TV = request.form.get("TV")
        Radio = request.form.get("Radio")
        Newspaper = request.form.get("Newspaper")

        # Create DataFrame based on input
        input_variables = pd.DataFrame([[TV, Radio, Newspaper]],
                                       columns=['TV', 'Radio', 'Newspaper'],
                                       dtype=float,
                                       index=['input'])
        # Get the model's prediction
        # Given that the prediction is stored in an array we simply extract by indexing
        prediction = model.predict(input_variables)[0]
        # We now pass on the input from the from and the prediction to the index page
        return render_template("index.html",
                                     original_input={'TV':TV,
                                                     'Radio':Radio,
                                                     'Newspaper':Newspaper},
                                     result=prediction
                                     )
    # If the request method is GET
    return render_template("index.html")
if __name__ == '__main__':
    app.run(debug=True)