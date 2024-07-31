from flask import Flask, render_template, jsonify
import json

app = Flask(__name__, template_folder='templates')

@app.route('/')
def index():
    return render_template('index.html')  # This will render a template from the templates folder

@app.route('/data')
def data():
    # Assuming your data is stored in 'data.json' in the static folder - default, since we didn't specify a static folder in the detectiongeoref_final.py 
    with open('/app/output/data.json') as f:
        data = json.load(f)
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)