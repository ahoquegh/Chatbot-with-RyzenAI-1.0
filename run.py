#
# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved. 
#
import argparse
import logging
import time
import gc
import os
import sys 
#sys.path.append("../ext") 
from ext.model_utils import warmup, decode_prompts, perplexity
from optimum.onnxruntime import ORTModelForCausalLM
from transformers import set_seed, AutoTokenizer, AutoTokenizer, OPTForCausalLM
import pathlib
import smooth
import torch
import random 
import string

CURRENT_DIR = pathlib.Path(__file__).parent
print(CURRENT_DIR.parent)
config_file_path = CURRENT_DIR / "vaip_config.json"

set_seed(123)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", help="Different OPT model sizes", type=str, default="opt-1.3b", choices=["opt-1.3b"])
    parser.add_argument("--download", help="load model from Huggingface and save it locally", action='store_true')
    parser.add_argument('--quantize', help="quantize model", action='store_true')
    parser.add_argument('--use_cache', help="Enable caching support", action='store_true')
    parser.add_argument("--local_path",help="Local directory path to ONNX model", default="")
    parser.add_argument("--target", help="cpu, aie", type=str, default="aie", choices=["cpu", "aie"])
    parser.add_argument('--disable_cache', help="Disable caching support", action='store_false')
    
    args = parser.parse_args()
    print(f"{args}")

    log_dir = "./logs_%s"%args.model_name
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = log_dir + "/log_%s_cpu.log"%(args.model_name)
    logging.basicConfig(filename=log_file,
                        filemode='w',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.CRITICAL)

    
    if args.download:        
        model = OPTForCausalLM.from_pretrained("facebook/" + args.model_name)
        out_dir = "./%s_pretrained_fp32"%args.model_name
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        model.save_pretrained(out_dir)
        print(f"Saving downloaded fp32 model...{out_dir}\n ")

    elif args.quantize:
        #---smooth quantize -----------------------
        path = "./%s_pretrained_fp32"%args.model_name
        if not os.path.exists(path):
            print(f"Pretrained fp32 model not found, exiting..")
            exit(1)
        model = OPTForCausalLM.from_pretrained(path)
        tokenizer = AutoTokenizer.from_pretrained("facebook/" + args.model_name)
        model.tokenizer = tokenizer 
        #print(f"Smooth quantiz the pretrained model...\n ")
        act_scales = torch.load(os.getenv("PYTORCH_AIE_PATH") + "/ext/smoothquant/act_scales/" + "%s.pt"%args.model_name)
        smooth.smooth_lm(model, act_scales, 0.5)
        print(model)
        
        prompt = ''.join(random.choices(string.ascii_lowercase + " ", k=model.config.max_position_embeddings))
        #inputs = tokenizer(prompt, return_tensors="pt")  # takes a lot of time
        inputs = tokenizer("What is meaning of life", return_tensors="pt") 
        print(f"inputs: {inputs}")
        print(f"inputs.input_ids: {inputs.input_ids}")
        for key in inputs.keys():
            print(inputs[key].shape)
            print(inputs[key])
        model_out = model(inputs.input_ids)
        print(f"{(model_out.logits.shape)=}")
        out_dir = "./%s_smoothquant"%args.model_name
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        model.save_pretrained(out_dir+"/model_onnx")
        print(f"Saving Smooth Quant fp32 model...\n ")

        #----------Quntize to int8--------------------------------
        print(f"Quantizing model with Optiaum...\n ")
        #proc = subprocess.Popen('cmd.exe', stdin = subprocess.PIPE, stdout = subprocess.PIPE)
        os.system('optimum-cli export onnx -m opt-1.3b_smoothquant\model_onnx --task text-generation-with-past opt-1.3b_smoothquant\model_onnx_int8  --framework pt --no-post-process')
        print(f"Saving quantized int8 model ...\n ")
    
    #-------Deploy and test model----------------------------------
    else: 
        if args.target == "aie":
            provider = "VitisAIExecutionProvider"
            provider_options = {'config_file': str(config_file_path)} 
        else:
            provider = "CPUExecutionProvider"
            provider_options = {} 
    
        path = "facebook/"
        if args.local_path != "":
            path = args.local_path
       
        model = ORTModelForCausalLM.from_pretrained(path, provider=provider,use_cache=args.disable_cache, use_io_binding=False, provider_options=provider_options)
        tokenizer = AutoTokenizer.from_pretrained("facebook/" +  args.model_name)

        collected = gc.collect()
    
        warmup(model, tokenizer, None)
        
        decode_prompts(model, tokenizer)
        logging.shutdown()