import requests
from requests.auth import HTTPBasicAuth

from CustomModel.CustomLLM import CustomLLM
from CustomModel.CustomEmbedding import CustomEmbedding
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
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

class RAG:
    def __init__(self):
        self.client_id = '<client_id>' # replace with your client_id
        self.client_secret = '<client_secret>' # replace with your client_secret
        self.userToken = None
        self.customLLM = None
        self.vectorStore = None

        set_llm_cache(SQLiteCache(database_path=".langchain.db"))

    def get_token(self):
        response = requests.get(
            '<url>/oauth/token?grant_type=client_credentials', # replace with your url
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
            url="<llm_api_url>", # replace with your llm_api_url
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
            url="<embedding_api_url>", # replace with your embedding_api_url
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
