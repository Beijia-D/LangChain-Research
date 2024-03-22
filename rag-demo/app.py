from flask import Flask, render_template, request, jsonify
import sys
sys.path.append("..")
from CustomModel import RAG
app = Flask(__name__)
ragPlus = RAG.RAGplus()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api', methods=['POST'])
def api():
    risk = request.json.get('risk_info')
    standard = request.json.get('risk_standard')
    result = ragPlus.call_your_api(
        risk,
        standard,
        path='Control.csv',
        collection_name='Control'
    )
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
