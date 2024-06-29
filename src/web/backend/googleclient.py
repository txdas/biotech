import os
my_api_key = "AIzaSyBOQOiFVm9q-Cj52RAr3jxBHzNjanCfvzk"
my_api_key = "AIzaSyBGbYe-20cwlwq2YAoUHuNDbb_sTo4ondg"
my_cse_id = "b1ba5999ddfb847fa"
OPENAI_API_KEY = "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"
os.environ["GOOGLE_CSE_ID"] = my_cse_id
os.environ["GOOGLE_API_KEY"] = my_api_key
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
from googleapiclient.discovery import build
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_google_community import GoogleSearchAPIWrapper
from langchain_community.retrievers.web_research import WebResearchRetriever
from langchain.chains import RetrievalQAWithSourcesChain

# Vectorstore
vectorstore = Chroma(embedding_function=OpenAIEmbeddings(),persist_directory="./data/chroma_db_oai")
# LLM
llm = ChatOpenAI(temperature=0)
# Search
search = GoogleSearchAPIWrapper()
search.k = 1
# Initialize
web_research_retriever = WebResearchRetriever.from_llm(
    vectorstore=vectorstore,
    llm=llm,
    search=search,
)
qa_chain = RetrievalQAWithSourcesChain.from_chain_type(llm, retriever=web_research_retriever)


def google_search(search_term, api_key, cse_id, **kwargs):
    os.environ["http_proxy"] = "http://127.0.0.1:33210"
    os.environ["https_proxy"] = "http://127.0.0.1:33210"
    service = build("customsearch", "v1", developerKey=api_key)
    res = service.cse().list(q=search_term, cx=cse_id, **kwargs).execute()
    return res


def chat_search(inputs):
    os.environ["http_proxy"] = "http://127.0.0.1:33210"
    os.environ["https_proxy"] = "http://127.0.0.1:33210"
    results = qa_chain({"question": inputs})
    return results["answer"]


def main():
    os.environ["http_proxy"] = "http://127.0.0.1:33210"
    os.environ["https_proxy"] = "http://127.0.0.1:33210"
    user_input = "How do LLM Powered Autonomous Agents work?"
    user_input = "如何设计一个优秀的ppt文档"
    # user_input = "治疗糖尿病有哪些常见的药物？"
    # user_input = "what are bio terms similar to envelope proteins?"
    result = chat_search(user_input)
    print(result)


def search():
    result = google_search("治疗糖尿病有哪些常见的药物", my_api_key, my_cse_id)
    for v in result["items"]:
        print(v)


if __name__ == '__main__':
    search()
