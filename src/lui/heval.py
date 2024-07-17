import gradio as gr
from openai_api import generate_code


def generate(description):
    return generate_code(description)


demo = gr.Interface(
    fn=generate,
    inputs=["text"],
    outputs=["text"],
)

demo.launch()