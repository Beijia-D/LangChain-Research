from flask import Flask, render_template, request, jsonify
import sys
import os
sys.path.append("..")
from CustomModel import RAG
app = Flask(__name__)
port = int(os.environ.get('PORT', 3001))
ragPlusPlus = RAG.RAGplus()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/getAIsuggestions', methods=['POST'])
def getAIsuggestions():
    risk = request.json.get('risk_info')
    standard = request.json.get('risk_standard')
    number = request.json.get('number')
    result = ragPlusPlus.simple_ai_suggestoin(
        risk,
        standard,
        number
    )
    return jsonify(result)

@app.route('/api', methods=['POST'])
def api():
    risk = request.json.get('risk_info')
    standard = request.json.get('risk_standard')
    result = ragPlusPlus.call_your_api(
        risk,
        standard,
        path='../data/Control.csv',
        collection_name='EMBEDDINGS_1'
    )
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port)
