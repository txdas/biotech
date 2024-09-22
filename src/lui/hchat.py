import time
import gradio as gr
from model_base import generate


def chat(message, history):
    response = generate(message)
    for i in range(len(response)):
        yield response[: i + 1]


demo = gr.ChatInterface(chat).queue()

if __name__ == "__main__":
    demo.launch()