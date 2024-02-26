from flask import Flask, render_template, request, jsonify
import requests
from requests.auth import HTTPBasicAuth
import csv
import pandas as pd

import sys
sys.path.append("..")
from CustomModel import CustomLLM

from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain
from langchain.schema import BaseOutputParser
from langchain.globals import set_llm_cache
from langchain.cache import SQLiteCache

import chromadb
set_llm_cache(SQLiteCache(database_path=".langchain.db"))

client_id = 'sb-75479b89-e2f4-4c3f-a722-486e9bdd9dc8!b39571|xsuaa_std!b77089'
client_secret = '2ae79f68-6785-4b28-8ed5-825e1ee154bf$nCq1GdPh6NeyfVNuIYtWP31bNcm-LEC6zmQIlboT6FU='
template = """You are an experienced risk assessment expert. Please utilize your knowledge of risk management and Security Frameworks like “ISO27001”, Privacy Frameworks like “GRDR”, “HIPAA” and other Compliance Frameworks to answer the questions based on the relevant information.
Please perform the following tasks:
1 - Understand Treatment type and treatment type classification standards.
   "Control": Control measures are designed to reduce the likelihood or impact of risks. They can include policies, procedures, guidelines, and physical or technological safeguards.
2 - Suggest approperiate treatments based on the risk user given. Your answer should be a list. Every record, it should only contain control name. “name” is the summary information of treatment, less than 10 words. Please use your knowledge to set name for your treatment.
ONLY return a comma separated list, and nothing more. For example: "name1,name2,name3,...nameN"
"""
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
human_template = "{risk}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
customLLM = None
chromadb_collection = None

class TransformToListFormat(BaseOutputParser):

    def parse(self, text: str):
        """Parse the output of an LLM call."""
        if text == "Unauthorized":
            return "Unauthorized"
        return text.strip().split(", ")

def get_token():
    response = requests.get(
        'https://learning.authentication.sap.hana.ondemand.com/oauth/token?grant_type=client_credentials',
        auth = HTTPBasicAuth(client_id, client_secret)
    )
    if (response.status_code != 200):
        print("Error: " + str(response.status_code))
        print(response.json())
        exit(1)
    return response.json()["access_token"]

def create_customLLM():
    userToken = get_token()
    return CustomLLM.CustomLLM(
        url="https://api.ai.internalprod.eu-central-1.aws.ml.hana.ondemand.com/v2/inference/deployments/d449d2c58f869ea0/chat/completions?api-version=2023-05-15",
        userToken=userToken,
        max_tokens=5000,
        temperature=0.0,
        frequency_penalty=0,
        presence_penalty=0
    )

def load_chromadb(path):
    global chromadb_collection
    id_list = []
    content_list = []
    metadata_list = []
    print("Start loading data...")
    with open(path, newline='') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=';')
        for row in reader:
            id_list.append(row['id'])
            content_list.append(row['name'] + ": " + row['description'])
            metadata_list.append({"significance": row['significance']})
    client = chromadb.Client()
    chromadb_collection = client.create_collection("available-controls")
    chromadb_collection.add(
        documents=content_list, # we handle tokenization, embedding, and indexing automatically. You can skip that and add your own embeddings as well
        metadatas=metadata_list, # filter on these!
        ids=id_list, # unique for each doc
    )
    print("End of loading data.")

def format_results(results):
    df = pd.read_csv('Control.csv', delimiter=';')

    id_list = []
    for result in results['ids']:
        id_list.extend(result)

    filtered_data = df[df['id'].isin(id_list)]
    return filtered_data.apply(lambda row: f"{row['id']}: {row['name']}", axis=1).str.cat(sep='\n')

def match_results(query_texts, n_results=1, where=None, where_document=None):
    if chromadb_collection is None:
        load_chromadb('Control.csv')
    print("Start matching results...")
    results = chromadb_collection.query(
        query_texts=query_texts,
        n_results=n_results,
        where=where,
        where_document=where_document
    )
    print("End of matching results.")
    return format_results(results)

app = Flask(__name__)

def call_your_api(input_data):
    global customLLM
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    if customLLM is None:
        print("Creating new customLLM...")
        customLLM = create_customLLM()
    chain = LLMChain(
        llm=customLLM,
        prompt=chat_prompt,
        output_parser=TransformToListFormat()
    )
    ai_results = chain.run(input_data)
    if ai_results == "Unauthorized":
        print("Token expired. Getting new token...")
        customLLM = create_customLLM()
        ai_results = chain.run(input_data)
    db_results = match_results(ai_results)
    return {'ai_suggestions': '\n>'.join(ai_results), 'db_suggestions': db_results}


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api', methods=['POST'])
def api():
    input_data = request.json.get('input_data')
    result = call_your_api(input_data)
    print(result)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
