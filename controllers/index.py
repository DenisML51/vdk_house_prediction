from app import app
from flask import render_template, request, session

@app.route('/', methods=['GET'])
def index():
    session['url'] = ''
    session['metrics'] = None
    session['pipeline'] = None
    session['prediction'] = None
    session['data'] = None
    session['metrics_train'] = None

    try:
        del session['models_data']
    except KeyError:
        pass

    return render_template(
        'index.html',
        len=len
    )


if __name__ == "__main__":
    app.run(debug=True)