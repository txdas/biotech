import requests
import json
import traceback
from io import StringIO

class UniprotClient(object):

    def __init__(self):
        self.base_url = "https://rest.uniprot.org/"
        pass

    def get(self, pid, db="uniprotkb", format="json"):
        try:
            url = f"{self.base_url}/{db}/{pid}"
            if format == "json":
                headers = {"accept": "application/json"}
                res = requests.get(url, headers=headers).text
                obj = json.loads(res)
                return obj
            elif format == "fasta":
                url = f"{url}?format=fasta"
                res = requests.get(url).text
                return res

        except:
            traceback.print_exc()
            pass

    def search(self, gene, organism, db="uniprotkb", size=1):
        try:
            query = f'"{organism}" AND (gene:{gene})'
            url = f"{self.base_url}/{db}/search?size={size}&query={query}"
            headers = {"accept": "application/json"}
            res = requests.get(url, headers=headers).text
            obj = json.loads(res)
            if "results" in obj and obj["results"]:
                return obj["results"][0]
        except:
            traceback.print_exc()
            pass