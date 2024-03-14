from flask import Flask, render_template, request, jsonify
import requests
from requests.auth import HTTPBasicAuth
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

set_llm_cache(SQLiteCache(database_path=".langchain.db"))

import os
from dotenv import load_dotenv
load_dotenv()

CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
AUTH_URL = os.getenv("AUTH_URL")
LLM_API_URL = os.getenv("LLM_API_URL")

template = """You are an experienced risk assessment expert. Please utilize your knowledge of risk management and Security Frameworks like “ISO27001”, Privacy Frameworks like “GRDR”, “HIPAA” and other Compliance Frameworks to answer the questions based on the relevant information.
Please perform the following tasks:
1 - Understand Treatment types and treatment type classification standards.
    1. "Control": Control measures are designed to reduce the likelihood or impact of risks. They can include policies, procedures, guidelines, and physical or technological safeguards.
    2. "Response": Response actions are the steps taken when a risk eventuates, or an incident occurs despite the control measures in place. These actions are intended to minimize the damage or harm caused by the risk.
2 - Suggest approperiate treatments based on the risk user given. Your answer should be a list. Every record, it should only contain type and name. “name” is the summary information of treatment, less than 10 words. Please use your knowledge to set name for your treatment.
The example of format: {format}
"""
response_format = '{"Type": <Type>,"Name": <name>},{"Type": <Type>,"Name": <name>}'
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
human_template = "{risk}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
customLLM = None

class TransformToJsonFormat(BaseOutputParser):

    def parse(self, text: str):
        """Parse the output of an LLM call."""
        if text == "Unauthorized":
            return "Unauthorized"
        new = {'result': text}
        return new

def get_token():
    response = requests.get(
        AUTH_URL,
        auth = HTTPBasicAuth(CLIENT_ID, CLIENT_SECRET)
    )
    if (response.status_code != 200):
        print("Error: " + str(response.status_code))
        print(response.json())
        exit(1)
    return response.json()["access_token"]

def create_customLLM():
    userToken = get_token()
    return CustomLLM.CustomLLM(
        url=LLM_API_URL,
        userToken=userToken,
        max_tokens=5000,
        temperature=0.0,
        frequency_penalty=0,
        presence_penalty=0
    )

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
        output_parser=TransformToJsonFormat()
    )
    result = chain.run(risk=input_data, format=response_format)
    if result == "Unauthorized":
        print("Token expired. Getting new token...")
        customLLM = create_customLLM()
        result = chain.run(risk=input_data, format=response_format)
    return result


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api', methods=['POST'])
def api():
    input_data = request.json.get('input_data')
    result = call_your_api(input_data)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
