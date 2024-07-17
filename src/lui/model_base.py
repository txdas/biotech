import transformers
from transformers import GemmaTokenizer, AutoModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer
print(transformers.__version__)


# model_id = "~/Desktop/code/notebook/models/codegemma-2b"
# tokenizer = GemmaTokenizer.from_pretrained(model_id)
# model = AutoModelForCausalLM.from_pretrained(model_id)
#
#
# def generate(input_text="Write me a Python function to calculate the nth fibonacci number."):
#     input_ids = tokenizer(input_text, return_tensors="pt")
#     outputs = model.generate(**input_ids, max_new_tokens=200)
#     return tokenizer.decode(outputs[0])


model = AutoModelForCausalLM.from_pretrained("./models/gpt2", local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained("./models/gpt2", local_files_only=True)

def generate(prompt="GPT2 is a model developed by OpenAI."):
#     prompt = "GPT2 is a model developed by OpenAI."

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    gen_tokens = model.generate(
        input_ids,
        do_sample=True,
        temperature=0.9,
        max_length=100,
    )
    gen_text = tokenizer.batch_decode(gen_tokens)[0]
    return gen_text