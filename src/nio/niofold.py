import requests
import json


class AlphaFold(object):

    def __init__(self):
        self.tlp = "https://alphafold.com/api/{}"

    def get(self, qualifier):
        url = self.tlp.format(f"prediction/{qualifier}")
        response = requests.get(url, verify=False,headers={"accept":"application/json"})
        lst = json.loads(response.text)
        return lst[0]


if __name__ == '__main__':
    af = AlphaFold()
    print(af.get("P00520"))