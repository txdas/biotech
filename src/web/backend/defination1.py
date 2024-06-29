from langchain_openai import ChatOpenAI
from langchain.schema import (
    HumanMessage,
    SystemMessage
)
from langchain.output_parsers import PydanticOutputParser
from backend import utils, outparser, dataloader, constant
import tqdm, os
OPENAI_API_KEY = "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"
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

chat = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
# chat = ChatOpenAI(temperature=0,model_name="gpt-4")
prompt_role = '''
        I want you act as biologist. 
    '''
prompt_init = '''
        You are a biologist. Here is the Envelop protein Definition,  
        #Envelop_Protein_Definition#: {definition}
        '''
definition_init = '''
        Envelope proteins are a group of proteins found in various viruses that are responsible for 
        facilitating viral entry into host cells and mediating viral attachment and fusion with host cell membranes, 
        including proteins that aid in the release and spread of the virus within the body.
'''

definition_init1 = '''
        Envelope proteins are a group of viral proteins that are located on the outer surface of the virus 
        and play a crucial role in viral entry, attachment, and fusion with host cells, 
        including facilitating the spread of the virus by assisting in the release of newly formed viruses to 
        infect other cells, such as through the cleavage of specific molecules on the host cell surface.'''

examples = '''
        coronaviruses --> Spike protein
        influenza viruses --> Hemagglutinin proteins
        Vesicular Stomatitis Virus --> Glycoprotein
        Paramyxoviridae Virus --> Fusion protein
        influenza viruses --> Neuraminidase proteins
'''

examples_ = '''
        Influenza Viruses  --> PB2-CAP
        Poliovirus --> 3D polymerase
        Human Immunodeficiency Virus --> reverse transcriptase
'''


def init_definition(name="Envelop Proteins", examples=examples):
    output_parser = PydanticOutputParser(pydantic_object=outparser.Definition)
    format_instructions = output_parser.get_format_instructions()
    name_ = "_".join(name.strip().split())
    prompt_define=f'''
        Your task is to generalize a definition of {name}.
        Given the following #Envelop Proteins#:
        {examples}
        Note that the #{name_}_Definition# should avoid the direct use of specific viruses or proteins, and not use examples.
        The #{name_}_Definition# focuses on a general description of the common characteristics and functions of these proteins. 
        The #{name_}_Definition# should exactly formulated in one sentence.
        {format_instructions}
    '''
    # The  # Envelop_Protein_Definition# should be in one sentence and don't use examples.
    messages = [
        SystemMessage(content=prompt_role),
        HumanMessage(content=prompt_define),
    ]
    output = chat(messages)
    result = output_parser.parse(output.content)
    print(result)
    return result.definition


def check(definition, virus, protein):
    messages = [
        SystemMessage(content=prompt_role),
        HumanMessage(content=f"Explain the function of {protein} in {virus} in short"
                             "the Explanation should be in one sentence and briefly"),
    ]
    output = chat(messages)
    # print(output)
    output_parser = PydanticOutputParser(pydantic_object=outparser.EnvelopDescProtein)
    format_instructions = output_parser.get_format_instructions()
    #  Explain the function of {protein} in {virus} briefly.
    question = f'''
        Does {protein} in {virus} conform to the #Envelop_Protein_Definition#?
        {format_instructions}
    '''
    message = prompt_init.format(definition=definition) + \
              "\n Classify the given following proteins whether conform to the #Envelop_Proteins_Definition#\n" + \
              output.content
    # print(output.content)
    # print(question)
    messages = [
        SystemMessage(content=message),
        HumanMessage(content=question),
    ]
    output = chat(messages)
    # print(output)
    result = output_parser.parse(output.content)
    return result


def modify_definition(definition, virus, protein, action="accept"):
    messages = [
        SystemMessage(content=prompt_role),
        HumanMessage(content=f"Explain the function of {protein} in {virus} in short"
                             "the Explanation should be in one sentence and briefly"),
    ]
    output = chat(messages)
    definition_parser = PydanticOutputParser(pydantic_object=outparser.Definition)
    format_instructions = definition_parser.get_format_instructions()
    prompt_modify = f'''
        Modify the #Envelop_Proteins_Definition# so that it could {action} {protein} in {virus}.
        The modified #Envelop_Proteins_Definition# must be abstractive and generalized.
        The modified #Envelop_Proteins_Definition# should avoid directly use of specific virus or proteins.
        The modified #Envelop_Proteins_Definition# should exactly formulated in one sentence and don't use examples。
        {format_instructions}
        '''
    message = prompt_init.format(definition=definition) + "\n" + output.content
    # print(message)
    messages = [
        SystemMessage(content=message),
        HumanMessage(content=prompt_modify)
    ]
    output = chat(messages)
    # print(output)
    result = definition_parser.parse(output.content)
    print(result)
    return result.definition


def main():
    # result = init_definition()
    # definition = result.definition
    definition = definition_init1
    # print(definition)
    total, hit = 0, 0
    number = utils.count(constant.HUMAN)
    bar = tqdm.tqdm(total=number)
    for i, v in enumerate(dataloader.load_human(limit=number)):
        # if i<100:
        #     continue
        # print(v)
        r = check(definition, v["organism"], v["protein"])
        flag = True if r.label == "Yes" else False
        total +=1
        if flag == v["target"]:
            hit += 1
        else:
            # action = "accept" if v["target"] else "reject"
            # definition = modify_definition(definition, v["organism"], v["protein"],action=action)
            print(v["organism"], v["protein"], v["target"], r.label)
        bar.update()
        print(hit, total, hit / total)


if __name__ == '__main__':
    # except: retrovirus matrix protein; Nipah virus Matrix protein; influenza viruses Neuraminidase proteins
    # definition_init = init_definition("Capping Enzyme",examples_)
    # modify_definition(definition_init, "influenza viruses", "Neuraminidase proteins") #
    # check(definition_init, "retrovirus", "matrix protein")
    main()
    '''
    marburgvirus matrix protein False Yes
    Influenza A virus Matrix protein False Yes
    Flavivirus M protein False Yes
    Avastrovirus VP25 True No
    '''


