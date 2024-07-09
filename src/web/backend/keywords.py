from langchain_openai import ChatOpenAI
from langchain.schema import (
    SystemMessage,
    HumanMessage
)
from backend  import outparser
from sortedcontainers import sortedset

chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
prompt_role = '''
    You are an pubmed search assistant.
    '''

define = '''
Envelop proteins are a class of viral proteins that are located on the surface of enveloped viruses and 
play a crucial role in viral entry, attachment, and fusion with host cells.
'''
keywords_name = '''
    You are an assistant tasked with improving Pubmed search results.
    Please generate more bio terms that are equal to {topic}.
    {format_instructions}
'''

keywords_function = '''
    Please generate search keywords for {topic} from the perspective of bio terms, 
    including molecular function, cellular component, or biological process.
    Each type keyword should be limit to three.
    {format_instructions}
'''


def keywords(topic="virus envelope proteins"):
    output_parser = outparser.MyJsonOutputParser(pydantic_object=outparser.SearchKeywords)
    format_instructions = output_parser.get_format_instructions()
    messages = [
        SystemMessage(content=prompt_role),
        HumanMessage(content=keywords_name.format(topic=topic,format_instructions=format_instructions)),
    ]
    output = chat(messages)
    result_name = output_parser.parse(output.content)
    output_parser = outparser.MyJsonOutputParser(pydantic_object=outparser.GOTermsKeywords)
    format_instructions = output_parser.get_format_instructions()
    messages.append(HumanMessage(content=keywords_function.format(topic=topic,format_instructions=format_instructions)))
    output = chat(messages)
    result_function = output_parser.parse(output.content)
    return result_name, result_function


def keywords_format(topic="virus envelope proteins"):
    result_name, result_function = keywords(topic)
    s = sortedset.SortedSet()
    lst = []
    for w in result_name.terms:
        if w not in s:
            s.add(w)
            lst.append(w)
    for w in result_function.molecular_function:
        if w not in s:
            s.add(w)
            lst.append(w)
    for w in result_function.cellular_component:
        if w not in s:
            s.add(w)
            lst.append(w)
    for w in result_function.biological_process:
        if w not in s:
            s.add(w)
            lst.append(w)
    print(lst)
    return "\n".join(lst)


if __name__ == '__main__':
    # keywords()
    keywords_format()