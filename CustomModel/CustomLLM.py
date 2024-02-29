from typing import Any, List, Mapping, Optional

import requests

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM

class CustomLLM(LLM):
    '''
    CustomLLM class for custom language model.
    This class is used to define a custom language model for the LangChain framework.
    It is a subclass of the LLM class from the langchain package
    '''

    url: str
    userToken: str
    max_tokens: int
    temperature: float
    frequency_penalty: float
    presence_penalty: float

    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        '''Get the response from SAP Generative AI Hub.'''
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        headers = {
            "Authorization": "Bearer " + self.userToken,
            "Content-Type": "application/json",
            "AI-Resource-Group": "default"
        }
        data = {
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "stop": "null"
        }
        print("Prompt: ", prompt)
        print("Sending request to get AI suggestions...")
        response = requests.post(
            self.url,
            json=data,
            headers=headers
        )
        if response.status_code == 401:
            return "Unauthorized"
        if response.status_code != 200:
            print(response)
            raise ValueError("Error in response")
        print("End of request...")
        print("Response: ", response.json())
        return response.json()["choices"][0]["message"]["content"]

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"url": self.url, "userToken": self.userToken, "max_tokens": self.max_tokens, "temperature": self.temperature, "frequency_penalty": self.frequency_penalty, "presence_penalty": self.presence_penalty}