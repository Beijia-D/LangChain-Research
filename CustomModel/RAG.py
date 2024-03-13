import requests
from requests.auth import HTTPBasicAuth
import json

from CustomModel.CustomLLM import CustomLLM
from CustomModel.CustomEmbedding import CustomEmbedding
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema import BaseOutputParser
from langchain.globals import set_llm_cache
from langchain.cache import SQLiteCache
from langchain_community.vectorstores import PGEmbedding
from langchain_community.document_loaders.csv_loader import CSVLoader

class TransformToListFormat(BaseOutputParser):

    def parse(self, text: str):
        """Parse the output of an LLM call."""
        if text == "Unauthorized":
            return "Unauthorized"
        return text.strip().split(", ")

class resultJsonFormat(BaseOutputParser):

    def parse(self, text: str):
        if text == "Unauthorized":
            return "Unauthorized"
        return json.loads(text.replace("\n", ""))

class RAG:
    def __init__(self):
        self.client_id = 'sb-75479b89-e2f4-4c3f-a722-486e9bdd9dc8!b39571|xsuaa_std!b77089'
        self.client_secret = '2ae79f68-6785-4b28-8ed5-825e1ee154bf$nCq1GdPh6NeyfVNuIYtWP31bNcm-LEC6zmQIlboT6FU='
        self.userToken = None
        self.customLLM = None
        self.vectorStore = None

        set_llm_cache(SQLiteCache(database_path=".langchain.db"))

    def get_token(self):
        response = requests.get(
            'https://learning.authentication.sap.hana.ondemand.com/oauth/token?grant_type=client_credentials',
            auth = HTTPBasicAuth(self.client_id, self.client_secret)
        )
        if response.status_code != 200:
            print("Error: " + str(response.status_code))
            print(response.json())
            exit(1)
        self.userToken = response.json()["access_token"]

    def create_customLLM(self):
        if self.userToken is None:
            self.get_token()
        return CustomLLM(
            url="https://api.ai.internalprod.eu-central-1.aws.ml.hana.ondemand.com/v2/inference/deployments/d449d2c58f869ea0/chat/completions?api-version=2023-05-15",
            userToken=self.userToken,
            max_tokens=5000,
            temperature=0.0,
            frequency_penalty=0,
            presence_penalty=0
        )

    def create_customEmbedding(self):
        if self.userToken is None:
            self.get_token()
        return CustomEmbedding(
            url="https://api.ai.internalprod.eu-central-1.aws.ml.hana.ondemand.com/v2/inference/deployments/dc04d48dce740753/embeddings?api-version=2023-05-15",
            userToken=self.userToken
        )

    def get_ai_suggestions(self, input_data):
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
        chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
        if self.customLLM is None:
            print("Creating new customLLM...")
            self.customLLM = self.create_customLLM()
        chain = LLMChain(
            llm=self.customLLM,
            prompt=chat_prompt,
            output_parser=TransformToListFormat()
        )
        ai_results = chain.run(input_data)
        if ai_results == "Unauthorized":
            print("Token expired. Getting new token...")
            self.get_token()
            self.customLLM = self.create_customLLM()
            ai_results = chain.run(input_data)
        return ai_results

    def load_documents(self, path, delimiter=';', quotechar='"', fieldnames=['id', 'name', 'description', 'significance'], source_column='id'):
        loader = CSVLoader(file_path=path, csv_args={
            'delimiter': delimiter,
            'quotechar': quotechar,
            'fieldnames': fieldnames
        }, source_column=source_column)
        return loader.load()

    def load_vectorstore(self, documents, collection_name, connection_string):
        self.customEmbedding = self.create_customEmbedding()
        self.vectorStore = PGEmbedding.from_documents(
            embedding=self.customEmbedding,
            documents=documents,
            collection_name=collection_name,
            connection_string=connection_string,
            pre_delete_collection=True
        )

    def get_db_suggestions(self, path, ai_results, collection_name, connection_string, k=2):
        if self.vectorStore is None:
            print("Loading vectorstore...")
            documents = self.load_documents(path)
            self.load_vectorstore(documents, collection_name, connection_string)
        page_contents = []
        print("Searching for similar documents...")
        for result in ai_results:
            for doc in self.vectorStore.similarity_search(result, k):
                page_contents.append(doc.page_content)
        unique_contents = list(set(page_contents))
        return '\n\n'.join(unique_contents)

    def call_your_api(self, input_data, path, collection_name, connection_string):
        ai_results = self.get_ai_suggestions(input_data)
        db_results = self.get_db_suggestions(
            path=path,
            ai_results=ai_results,
            collection_name=collection_name,
            connection_string=connection_string
        )
        
        return {'ai_suggestions': '\n'.join(ai_results), 'db_suggestions': db_results}

class RAGplus:
    def __init__(self):
        self.client_id = 'sb-75479b89-e2f4-4c3f-a722-486e9bdd9dc8!b39571|xsuaa_std!b77089'
        self.client_secret = '2ae79f68-6785-4b28-8ed5-825e1ee154bf$nCq1GdPh6NeyfVNuIYtWP31bNcm-LEC6zmQIlboT6FU='
        self.userToken = None
        self.customLLM = None
        self.vectorStore = None

        set_llm_cache(SQLiteCache(database_path=".langchain.db"))

    def get_token(self):
        response = requests.get(
            'https://learning.authentication.sap.hana.ondemand.com/oauth/token?grant_type=client_credentials',
            auth = HTTPBasicAuth(self.client_id, self.client_secret)
        )
        if response.status_code != 200:
            print("Error: " + str(response.status_code))
            print(response.json())
            exit(1)
        self.userToken = response.json()["access_token"]

    def create_customLLM(self):
        if self.userToken is None:
            self.get_token()
        return CustomLLM(
            url="https://api.ai.internalprod.eu-central-1.aws.ml.hana.ondemand.com/v2/inference/deployments/d449d2c58f869ea0/chat/completions?api-version=2023-05-15",
            userToken=self.userToken,
            max_tokens=5000,
            temperature=0.0,
            frequency_penalty=0,
            presence_penalty=0
        )

    def create_customEmbedding(self):
        if self.userToken is None:
            self.get_token()
        return CustomEmbedding(
            url="https://api.ai.internalprod.eu-central-1.aws.ml.hana.ondemand.com/v2/inference/deployments/dc04d48dce740753/embeddings?api-version=2023-05-15",
            userToken=self.userToken
        )

    def get_ai_suggestions(self, risk, standard, controls):
        template = """
        You are an experienced risk assessment expert.There are many widely used risk management standards: ISO 31000, COSO ERM, NIST SP 800-30, and ISO/IEC 27005.
        You need to know that control measures are designed to reduce the likelihood or impact of risks. They can include policies, procedures, guidelines, and physical or technological safeguards.
        Now, users have given you the risk information: {risk}. Users want to get some suggestions for controls according to the risk management standard {standard}. Here we found some existing controls for you:
        {controls}.
        Above controls may be empty or may not be applicable for the risk.
        Please first check if there are any controls that meet the standard and are applicable to the risk provided by the user. If no specific controls were provided in the prompt, just give some suggestions. If there are applicable controls, please retain these controls. You can also suggest additional control measures.
        ONLY return a dictionary, containing two keys: "applicable_controls" and "suggested_controls". Each key corresponds to a list as its value. The list can be empty. The items in the list is also in dictionary, contains three keys: "id", "name", "description". Don't add anything more in the response.
        """
        if self.customLLM is None:
            print("Creating new customLLM...")
            self.customLLM = self.create_customLLM()
        prompt = PromptTemplate.from_template(template)
        chain = LLMChain(
            llm=self.customLLM,
            prompt=prompt,
            output_parser=resultJsonFormat()
        )
        ai_results = chain.run(risk=risk, standard=standard, controls=controls)
        if ai_results == "Unauthorized":
            print("Token expired. Getting new token...")
            self.get_token()
            self.customLLM = self.create_customLLM()
            ai_results = chain.run(risk=risk, standard=standard, controls=controls)
        return ai_results

    def load_documents(self, path, delimiter=';', quotechar='"', fieldnames=['id', 'name', 'description', 'significance'], source_column='id'):
        loader = CSVLoader(file_path=path, csv_args={
            'delimiter': delimiter,
            'quotechar': quotechar,
            'fieldnames': fieldnames
        }, source_column=source_column)
        return loader.load()

    def load_vectorstore(self, documents, collection_name, connection_string):
        self.customEmbedding = self.create_customEmbedding()
        self.vectorStore = PGEmbedding.from_documents(
            embedding=self.customEmbedding,
            documents=documents,
            collection_name=collection_name,
            connection_string=connection_string,
            pre_delete_collection=True
        )

    def get_db_suggestions(self, path, query, collection_name, connection_string, k=10):
        if self.vectorStore is None:
            print("Loading vectorstore...")
            documents = self.load_documents(path)
            self.load_vectorstore(documents, collection_name, connection_string)
        relevant_docs = []
        print("Searching for relevant documents...")
        for doc, score in self.vectorStore.similarity_search_with_score(query, k=k):
            if score < 0.65:
                relevant_docs.append(doc.page_content)
        print("relevant controls:", relevant_docs)
        return '\n'.join(relevant_docs)

    def call_your_api(self, risk, standard, path, collection_name, connection_string):
        relevant_controls = self.get_db_suggestions(
            path=path,
            query=risk,
            collection_name=collection_name,
            connection_string=connection_string
        )
        ai_results = self.get_ai_suggestions(risk, standard, relevant_controls)
        result = {'ai_suggestions': "The AI did not provide additional controls.", 'db_suggestions': "No applicable controls found."}
        if len(ai_results['applicable_controls'])>0:
            result['db_suggestions']="\n\n".join([f"ID: {item['id']}\nName: {item['name']}\nDescription: {item['description']}" for item in ai_results['applicable_controls']])
        if len(ai_results['suggested_controls'])>0:
            result['ai_suggestions']="\n\n".join([f"Name: {item['name']}\nDescription: {item['description']}" for item in ai_results['suggested_controls']])
        
        return result