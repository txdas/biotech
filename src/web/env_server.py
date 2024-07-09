import warnings, sys
warnings.filterwarnings('ignore')
sys.path.append("./")
import gradio as gr
from backend.defination1 import init_definition, modify_definition, check
from backend.keywords import keywords_format
from backend.googleclient import chat_search

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
define = """
Envelop proteins are a group of viral proteins that are located on the outer surface of the virus and play a crucial role in viral entry, attachment, fusion, and other important processes
"""
# outputs=[
#     gr.HighlightedText(label="Output")
# ],
# share=True


def check_protein(definition, virus, protein):
    res = check(definition, virus, protein)
    message = res.label + "," + res.explanation
    return message

app0 = gr.Interface(
    fn=keywords_format,
    inputs=[
        gr.Textbox(label="Topic")
    ],
    outputs=gr.Textbox(label="Keywords"),
    title="Keywords",
    examples=[["Virus Envelop Proteins"], ["Capping Enzyme"], ["protease enzyme"]]
)

app1 = gr.Interface(
    fn=init_definition,
    inputs=[
        gr.Textbox(label="Name"),
        gr.components.Textbox(lines=2, label="Examples", placeholder="none"),
    ],
    outputs=gr.Textbox(label="Definition"),
    title = "Init Definition",
    examples=[["Envelop Protein", examples], ["Capping Enzyme", examples_]]
)
app2 = gr.Interface(
    fn=check_protein,
    inputs=[
        gr.components.Textbox(lines=2, label="Input", placeholder="none"),
        gr.Textbox(label="Virus"),
        gr.Textbox(label="Protein")
    ],
    outputs=gr.Textbox(label="Result"),
    title="Check Definition",
    examples=[[define, "influenza viruses", "Neuraminidase proteins"]]
)

app3 = gr.Interface(
    fn=modify_definition,
    inputs=[
        gr.components.Textbox(lines=2, label="Input", placeholder="none"),
        gr.Textbox(label="Virus"),
        gr.Textbox(label="Protein"),
        gr.Dropdown(['accept', 'reject'])
    ],
    outputs=gr.Textbox(label="Modify Definition"),
    title="Modify Definition",
    examples=[[define, "influenza viruses", "Neuraminidase proteins", "accept"]]
)

app4 = gr.Interface(
    fn=chat_search,
    inputs=[
        gr.Textbox(label="Question")
    ],
    outputs=gr.Textbox(label="Answer"),
    title="Chat search",
    examples=[["如何设计一个优秀的ppt文档?"], ["治疗糖尿病有哪些常见的药物？"],
              ["some biology terms which is equal to viral Envelope Proteins"]]
)


demo = gr.TabbedInterface([app0, app1, app2, app3, app4],
                          ["Keywords","Init Definition", "Check Definition", "Modify Definition", "Chat search"])

# demo.launch(share=True, server_port=8081, server_name="127.0.0.1")
demo.launch(server_port=8081)

