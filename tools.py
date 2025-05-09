from langchain.tools import BaseTool
from typing import Type, Optional
from pydantic import Field
import math
import requests

class CalculatorTool(BaseTool):
    name: str = "Calculator"
    description: str = "Performs basic math calculations based on a user query."

    def _run(self, query: str) -> str:
        try:
            # Safe eval context
            allowed_names = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
            result = eval(query, {"__builtins__": {}}, allowed_names)
            return str(result)
        except Exception as e:
            return f"Error in calculation: {e}"

    def _arun(self, query: str):
        raise NotImplementedError("Async not supported.")


class DefineTool(BaseTool):
    name: str = "Define"
    description: str = "Returns a simple definition for a term using Free Dictionary API."

    def _run(self, query: str) -> str:
        word = query.strip().lower()
        url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}"

        try:
            response = requests.get(url)
            if response.status_code != 200:
                return f"No definition found for '{word}'."

            data = response.json()
            if isinstance(data, list) and "meanings" in data[0]:
                meanings = data[0]["meanings"]
                definitions = meanings[0]["definitions"]
                first_def = definitions[0]["definition"]
                return f"{word.capitalize()}: {first_def}"
            else:
                return f"No clear definition found for '{word}'."
        except Exception as e:
            return f"Error fetching definition: {e}"

    def _arun(self, query: str):
        raise NotImplementedError("Async not supported.")
