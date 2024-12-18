from app import app
from flask import render_template, request, session
from models.parser_model import parser_1
import pandas as pd


@app.route('/data')
def data():
    df = pd.read_csv('C:\\Coding\\vdk_prices - Copy\\data_3.csv')

    head_data = df.head(5).to_html()
    describe_data = df.describe().to_html()

    return render_template('data.html', head_data=head_data, describe_data=describe_data)


if __name__ == '__main__':
    app.run(debug=True)

