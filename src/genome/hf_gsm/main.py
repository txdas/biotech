from config import GlmConfig
from tokenizer import GlmTokenizer
from model_base import GlmModel,GlmForMaskedLM
from torchinfo import summary

if __name__ == '__main__':
    conf = GlmConfig().from_pretrained("config")
    tokenizer = GlmTokenizer(vocab_file="./config/vocab.txt")
    # print(conf.to_dict())
    encoded_input = tokenizer("ACGTACC",max_length=128,truncation=True,return_tensors='pt')
    print(encoded_input)
    # model = GlmForMaskedLM(conf)
    # output = model(**encoded_input)
    # print(output)
    # print(summary(model, depth=5, input_size=(2, 512), dtypes=['torch.IntTensor']))


