import requests
from requests.auth import HTTPBasicAuth
class ControlRetriever:
    def __init__(self, credentials):
        self.credentials = credentials
        self.token = None
    def get_token(self):
        print("Getting token for control api")
        response = requests.get(
            self.credentials["url"] + "/oauth/token?grant_type=client_credentials",
            auth = HTTPBasicAuth(self.credentials["clientid"], self.credentials["clientsecret"])
        )
        if response.status_code != 200:
            print("Error: " + str(response.status_code))
            print(response.json())
            exit(1)
        self.token = response.json()["access_token"]

    def format(self, controls):
        print("Formatting controls")
        control_texts = []
        metadatas = []
        for control in controls:
            control_texts.append(f"ID: {control['displayId']}\nName: {control['controlName']}\nDescription: {control['description']}")
            metadatas.append({"source": control['displayId']})
        return [control_texts, metadatas]
    
    def get_controls(self):
        if self.token is None:
            self.get_token()
        print("Getting controls")
        response = requests.get(
            self.credentials["endpoints"] + '/odata/v4/ComplianceControlService/ComplianceControl',
            headers = {
                "Authorization": "Bearer " + self.token
            }
        )
        if response.status_code == 401:
            self.get_token()
        if response.status_code != 200:
            print(response)
            raise ValueError("Error in response")
        return self.format(response.json()["value"])