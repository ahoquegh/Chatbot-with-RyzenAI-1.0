import torch
import logging 
import time 

import random
random.seed(0)
import numpy as np 

def get_wikitext2(tokenizer, dataset="non-raw", nsamples=128, seqlen=2048):
    """ gptq """
    from datasets import load_dataset
    if dataset == "non-raw":
        traindata = load_dataset('wikitext', 'wikitext-2-v1', split='train')
        testdata = load_dataset('wikitext', 'wikitext-2-v1', split='test')
    elif dataset == "raw":
        traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
        testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    else:
        raise ValueError(
                "You are using an unsupported dataset, only support wikitext2-raw-v1 and wikitext2-v1."
                "Using wikitext2-raw-v1 with --dataset=raw and wikitext2-v1 with --dataset=non-raw."
            )

    trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')
    dataloader = []
    for _ in range(nsamples):
        i = random.randint(0, testenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = testenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        dataloader.append((inp, tar))
    return dataloader, testenc

def benchmark(model, input_ids):
    """ from gptq """
    cache = {'past': None}
    def clear_past(i):
        def tmp(layer, inp, out):
            if cache['past']:
                cache['past'][i] = None
        return tmp
    for i, layer in enumerate(model.model.decoder.layers):
        layer.register_forward_hook(clear_past(i))
    tot = 0.0
    with torch.no_grad():
        attention_mask = torch.ones((1, input_ids.numel()))
        times = []
        for i in range(input_ids.numel()):
            tick = time.time()
            out = model(
                input_ids[:, i].reshape((1,-1)),
                past_key_values=cache['past'],
                attention_mask=attention_mask[:, :(i + 1)].reshape((1, -1))
            )
            times.append(time.time() - tick)
            tot += times[-1]
            #print(i, times[-1], out.logits.shape)
            cache['past'] = list(out.past_key_values)
            del out
        import numpy as np
        #print('Median:', np.median(times))
    return np.median(times), tot

def perplexity(model, tokenizer, dataset):
    random.seed(0)
    np.random.seed(0)
    torch.random.manual_seed(0)
    print(f"Calculating Perplexity on wikitext2 test set ...")
    model = model#.cuda()
    dataloader, testenc = get_wikitext2(tokenizer, dataset=dataset)
    
    model.seqlen = model.config.max_position_embeddings #2048
    test_enc = testenc.input_ids
    nsamples = test_enc.numel() // model.seqlen
    dtype = next(iter(model.parameters())).dtype

    cache = {'past': None}
    def clear_past(i):
        def tmp(layer, inp, out):
            if cache['past']:
                cache['past'][i] = None
        return tmp
    for i, layer in enumerate(model.model.decoder.layers):
        layer.register_forward_hook(clear_past(i))

    loss = torch.nn.CrossEntropyLoss()
    nlls = []

    with torch.no_grad():
        attention_mask = torch.ones((1, test_enc.numel()))#.cuda()
        
        for i in range(nsamples):
            batch = test_enc[:, (i * model.seqlen):((i + 1) * model.seqlen)]#.cuda()
            out = model(
                batch,
                attention_mask=attention_mask[:, (i * model.seqlen):((i + 1) * model.seqlen)].reshape((1, -1))
            )
            shift_labels = test_enc[
                :, (i * model.seqlen):((i + 1) * model.seqlen)
            ][:, 1:]#.cuda()
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(out.logits[0][:-1, :], shift_labels.view(-1))
            neg_log_likelihood = loss.float() * model.seqlen
            nlls.append(neg_log_likelihood)

        ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
        print('Perplexity:', ppl.item())


prompts = []

org_prompts = [ "What is the meaning of life?",             
            "Tell me something you don't know.",        
            "What does Xilinx do?",                     
            "What is the mass of earth?",                
            "What is a poem?",                          
            "What is recursion?",                        
            "Tell me a one line joke.",                  
            "Who is Gilgamesh?",                         
            "Tell me something about cryptocurrency.",  
            "How did it all begin?"                     
            ]

def warmup(model, tokenizer, quant_mode):
    print("*"*10)
    print("Tesing the Model...")
    print("*"*10)
    prompt = "What does AMD do?"
    
    start = time.time()
    inputs = tokenizer(prompt, return_tensors="pt") 
    end = time.time()
    logging.critical(f"[PROFILE][WARMUP] tokenizer: {end-start}")

    start = time.time()
    generate_ids = model.generate(inputs.input_ids, max_length=30)
    end = time.time()
    logging.critical(f"[PROFILE][WARMUP] generate: {end-start} .. for 30 tokens.")

    start = time.time()
    response = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    end = time.time()
    logging.critical(f"[PROFILE][WARMUP] tokenizer decode: {end-start}")
    
    print(response)
    
def decode_prompts(model, tokenizer):
    for prompt in prompts:
        logging.critical("*"*10)
        print("*"*10)
        start = time.time()
        inputs = tokenizer(prompt, return_tensors="pt") 
        end = time.time()
        logging.critical(f"[PROFILE][CPU] tokenizer: {end-start}")
        #print(f"[PROFILE][CPU] tokenizer: {end-start}")

        start = time.time()
        generate_ids = model.generate(inputs.input_ids, max_length=30)
        end = time.time()
        logging.critical(f"[PROFILE][AIE] generate: {end-start} .. for 30 tokens.")
        #print(f"[PROFILE][AIE] generate: {end-start} .. for 30 tokens.")

        start = time.time()
        response = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        end = time.time()
        logging.critical(f"[PROFILE][CPU] tokenizer decode: {end-start}")
        #print(f"[PROFILE][CPU] tokenizer decode: {end-start}")
        
        print(response)
        logging.critical(f"response: {response}")