import gradio as gr
import time
import torch
import os
from transformers import pipeline, AutoTokenizer, set_seed
from modeling_ort_amd import ORTModelForCausalLM
import argparse 
from utils import Utils
import gc 
import smooth
import os
import pathlib
#import qlinear 

def load_model(model_path, model_file):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print("Load FP32 model...")
    model = torch.load(model_file)
    model.eval()

    node_args = ()
    node_kwargs = {}
    print("Deploy smooth quant...")
    #Utils.replace_node( model, 
    #                    torch.ao.nn.quantized.dynamic.modules.linear.Linear,
    #                    qlinear.QLinear, 
    #                    node_args, node_kwargs 
    #                  )
    collected = gc.collect()
    return model, tokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--model_file", help="quantized_model_file with *.pth extension", type=str)
args = parser.parse_args()

set_seed(123)
#-------------------alim -----------------
#model, tokenizer= load_model("facebook/opt-1.3b",args.model_file)
CURRENT_DIR = pathlib.Path(__file__).parent
print("Currnt DIR: " ,CURRENT_DIR.parent)
config_file_path = CURRENT_DIR.parent / "vaip_config.json"
print("config_file_path: ", config_file_path)
provider = "VitisAIExecutionProvider"
provider_options = {'config_file': str(config_file_path)} 
model = ORTModelForCausalLM.from_pretrained(args.model_file, provider=provider,use_cache=True, use_io_binding=False, provider_options=provider_options)
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")
#model.eval()
#--------------------

def generate_text(Input_text, max_output_token):
    inputs = tokenizer(Input_text, return_tensors="pt") 
    s = time.perf_counter()
    outputs_tkn = model.generate(inputs.input_ids, max_length=max_output_token, use_cache=True, do_sample=False)
    e = time.perf_counter() - s
    outputs_tkn_len = outputs_tkn.shape[1]
    outputs = tokenizer.batch_decode(outputs_tkn,
                                    skip_special_tokens=True, 
                                    clean_up_tokenization_spaces=False)[0]
    print(outputs)
    yield outputs, "tkn/sec: " + str(outputs_tkn_len/e)

gr.Interface(
    fn=generate_text,
    inputs=["text", gr.Slider(minimum=32, maximum=256, value=32, step = 32)],
    outputs=["text", "text"],
    title="Chatbot on Ryzen AI",
    description="Simple opt chatbot on Ryzen AI Laptop",
    concurrency_limit=4
    #).queue(concurrency_count=2).launch(server_name="localhost", server_port=1234)
    ).queue().launch(server_name="localhost", server_port=1234)
