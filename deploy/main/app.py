from flask import Flask, render_template, request, jsonify
from cfenv import AppEnv
from hdbcli import dbapi

import os
port = int(os.environ.get('PORT', 3001))

import sys
sys.path.append("..")
from CustomModel import RAG

app = Flask(__name__)
env = AppEnv()
hana = env.get_service(label="hana")
aicore = env.get_service(label="aicore")
db_connection = None
ai_credentials = None
if hana is None:
    raise ValueError("No HANA service bound to the application.")
else:
    db_connection = dbapi.connect(address=hana.credentials['host'],
            port=int(hana.credentials['port']),
            user=hana.credentials['user'],
            password=hana.credentials['password'],
            encrypt='true',
            sslTrustStore=hana.credentials['certificate'])

if aicore is None:
    raise ValueError("No AI Core service bound to the application.")
else:
    ai_credentials = aicore.credentials

ragPlusPlus = RAG.RAGplus(ai_credentials, db_connection)

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
    app.run(host='0.0.0.0', port=port, debug=True)
