from flask import Flask, render_template, request, jsonify
import sys
sys.path.append("..")
from CustomModel import RAG
app = Flask(__name__)
rag = RAG.RAG()
ragPlus = RAG.RAGplus()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api', methods=['POST'])
def api():
    # input_data = request.json.get('input_data')
    # result = rag.call_your_api(
    #     input_data,
    #     path='Control.csv',
    #     collection_name='Control'
    # )
    # print(result)
    # return jsonify(result)
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
