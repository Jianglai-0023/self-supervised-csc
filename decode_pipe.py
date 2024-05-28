from __future__ import absolute_import, division, print_function
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, IterableDataset
from tqdm.auto import tqdm, trange
from transformers import AutoTokenizer
from accelerate import Accelerator
from autocsc import *
from run_mlm import *
from UD import VocabProcessor
from typing import Optional, List
import torch
import datetime 

class InputExample(object):
    def __init__(self,src, trg):
        self.src = src
        self.trg = trg

def stdf(string):
    def _h(char):
        inside_code = ord(char)
        if inside_code == 0x3000:
            inside_code = 0x0020
        else:
            inside_code -= 0xfee0
        if inside_code < 0x0020 or inside_code > 0x7e:
            return char
        return chr(inside_code)
    return "".join([_h(char) for char in string])

def _read(input_file):
        with open(input_file, "r", encoding="utf-8") as f:
            lines = []
            for line in f:
                src, trg = line.strip().split("\t")
                src, trg = stdf(src), stdf(trg)
                if len(src.split())==1:
                    src = " ".join(src)
                    trg = " ".join(trg)
                lines.append((src.split(), trg.split()))
            return lines
        
def _create_examples(lines):
        examples = []
        for i, (src, trg) in enumerate(lines):
            if len(src) == len(trg):         
                    examples.append(InputExample(src=src, trg=trg))
        return examples

def customize(top_k_scores, log_topk_scores, topk_indexes,processor,tokenizer,k_size,length,weight = 4):
    max_logit = 1
    min_logit = 2e-5
    # todo not sure about logit
    res_sent = []
    for j in range(length):
        res_token = []
        for k in range(k_size):
            if top_k_scores[j][k] < min_logit and k != 0:
                break
            res_token.append([log_topk_scores[j][k], topk_indexes[j][k]])
            if top_k_scores[j][k] >= max_logit:
                break
        res_sent.append(res_token)

    res_sents = []
    def global_search(sent, t):
        if len(t) == 0:
            res_sents.append(list(sent.strip().split(" ")))
        else:
            for j in t[0]:
                global_search(f"{sent} {j[0]},{j[1]}", t[1:])

   
    global_search("", res_sent)

    res_sents_ids = [[int(y.split(",")[1]) for y in x] for x in res_sents]
    res_sents_scores = [[float(y.split(",")[0]) for y in x] for x in res_sents]

    res_sents = [[tokenizer.convert_ids_to_tokens(y) for y in x] for x in res_sents_ids]

    # calculate the additional score
    vocab_len = [processor.get_vocab_length("".join(tokens)) for tokens in res_sents]
    vocab_mean = np.mean(vocab_len)
    vocab_len = [(x - vocab_mean) for x in vocab_len]
    scores = [weight * v_l + sum(score) for v_l, score in zip(vocab_len, res_sents_scores)]

    largest_prob = 0
    for k in range(len(scores)):
        if scores[largest_prob] < scores[k]:
            largest_prob = k
    res_sent_id = res_sents_ids[largest_prob]
    res_sent = res_sents[largest_prob]
    

    return res_sent_id, res_sent

def main():
    parser = argparse.ArgumentParser()
    # Data config
    parser.add_argument("--load_state_dict", type=str, default="")
    parser.add_argument("--eval_on", type=str, default="")
    parser.add_argument("--vocab", type=str, default="wordlist/医学.txt")
    parser.add_argument("--load_model_path", type=str, default="bert-base-chinese")
    parser.add_argument("--max_seq_length", type=int, default=128)
    parser.add_argument("--cache_dir", type=str, default="../cache/")
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--calc_prd", action="store_true") 
    parser.add_argument("--data_path", type=str, default="/home/jiangl/csc")
    parser.add_argument("--pick_max", action="store_true")
    parser.add_argument("--change_logits", type=float, default=0.9)
    parser.add_argument("--max_range", type=int, default=20)
    parser.add_argument("--use_ud", action="store_true")
    parser.add_argument("--k_size", type=int, default=5)
    parser.add_argument("--accelerate", action="store_true")
    parser.add_argument("--acc_range", type=int, default=8)

    args = parser.parse_args() 
    startt = datetime.datetime.now()
    # args = args_item()
    vocab_filename = args.vocab
    processor = VocabProcessor(vocab_filename)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    i = 0
    output_result_file = os.path.join(args.output_dir,"sents.result")
    
    model = AutoCSCReLM.from_pretrained(args.load_model_path,
                                    cache_dir="../cache")
    model.load_state_dict(torch.load(args.load_state_dict), strict=False)
    model.to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.load_model_path,
                                          cache_dir="../cache")
    model.eval()
    # accelerator = Accelerator(cpu=False, mixed_precision="fp16")
    data_path = os.path.join(args.data_path,args.eval_on)
    data = _create_examples(_read(data_path))
    all_inputs,all_trgs,all_prd = [],[],[]

    def decode(input_ids):
        return tokenizer.convert_ids_to_tokens(input_ids, skip_special_tokens=True)

    def calc_prd(src_ids_,is_masked = False,ori = None,mask_idx = None):

        src_ids = src_ids_[:]
        src_ids = [tokenizer.cls_token_id] + src_ids + [tokenizer.sep_token_id] + [tokenizer.mask_token_id for _ in src_ids] + [tokenizer.sep_token_id]
        # attention_mask = [1] * len(src_ids)
        src_ids = torch.LongTensor([src_ids]).to("cuda")
        # attention_mask = torch.LongTensor([attention_mask]).to("cuda") 
        outputs = model(src_ids)
        logits = outputs["logits"]
        prd_ids = outputs["predict_ids"]

        probs = model.softmax(logits)
        flag = False
        if ori is not None:
            prd_rank_prob, prd_rank_ids = torch.topk(probs,k=args.max_range,dim=-1,largest=True,sorted=True)
            rank = 0    
            for c in prd_rank_ids[0][mask_idx + 1]:
                rank += 1
                if rank < args.max_range:
                    if c == ori:
                        flag = True
                else: break
                # if rank == 10:
                #     break

        prd_probs, prd_ids = torch.max(probs, -1)
        _p = [pt for pt, st in zip(prd_ids[0], src_ids[0]) if st == tokenizer.mask_token_id]
        _pp = [pt for pt, st in zip(prd_probs[0], src_ids[0]) if st == tokenizer.mask_token_id]
        if is_masked:
            _p = _p[1:]
            _pp = _pp[1:]
            if flag:
                _p[mask_idx] = ori
        return _p,_pp
    
    def calc_topk(input_ids,k_size):
        src_ids = input_ids[:]
        src_ids = [tokenizer.cls_token_id] + src_ids + [tokenizer.sep_token_id] + [tokenizer.mask_token_id for _ in src_ids] + [tokenizer.sep_token_id]
        
        src_ids = torch.LongTensor([src_ids]).to("cuda")
        
        outputs = model(src_ids)
        logits = outputs["logits"]
        prd_ids = outputs["predict_ids"]
        probs = nn.LogSoftmax(dim=-1)(logits)
        log_topk_scores, topk_indexes = probs.topk(k=k_size, dim=-1, sorted=True, largest=True)
        
        top_k_scores, log_topk_scores, topk_indexes = torch.exp(log_topk_scores).tolist(), log_topk_scores.tolist(), topk_indexes.tolist()
        
        top_k_scores = [sco for sco,s in zip(top_k_scores[0],src_ids[0]) if s == tokenizer.mask_token_id] 
        log_topk_scores = [sco for sco,s in zip(log_topk_scores[0],src_ids[0]) if s == tokenizer.mask_token_id] 
        topk_indexes = [sco for sco,s in zip(topk_indexes[0],src_ids[0]) if s == tokenizer.mask_token_id] 
        
        return top_k_scores, log_topk_scores, topk_indexes 


    def customize(top_k_scores,log_topk_scores, topk_indexes,processor,tokenizer,k_size,length,weight = 4):
    
        max_logit = 1
        min_logit = 2e-5
        # todo not sure about logit
        res_sent = []
        for j in range(length):
            res_token = []

            for k in range(k_size):
            
                if top_k_scores[j][k] < min_logit and k != 0:
                
                    break
                res_token.append([log_topk_scores[j][k], topk_indexes[j][k]])
                if top_k_scores[j][k] >= max_logit:
                
                    break
            res_sent.append(res_token)

        res_sents = []

        def global_search(sent, t):
            # print(t)
            if len(t) == 0:
                res_sents.append(list(sent.strip().split(" ")))
            else:
                for j in t[0]:
                    # print(t[0])
                    global_search(f"{sent} {j[0]},{j[1]}", t[1:])

        def convert_type(mid_results: List):
            mid_results = [res[:-1] for res in mid_results]
            output = []
            for res in mid_results:
                result = []
                for score, token in res:
                    result.append(str(score) + ',' + str(token))
                output.append(result)
            return output

        def beam_search(res_sent, beam_size=20):
            mid_results = []
            results = []
            for res in res_sent:
                if not mid_results:
                    for i in range(len(res)):
                        temp_candidate = []
                        temp_candidate.append(res[i])
                        temp_candidate.append(res[i][0])
                        mid_results.append(temp_candidate)
                else:
                    results = []
                    for result in mid_results:
                        total_score = result[-1]
                        for score, token in res:
                            new_result = result[:-1] + [[score, token]] + [total_score + score]
                            results.append(new_result)
                    mid_results = results.copy()
                    # convert
                    output = convert_type(mid_results)
                    # print(output)
                    # reward
                    res_sents_ids = [[int(y.split(",")[1]) for y in x] for x in output]
                    res_sents_scores = [[float(y.split(",")[0]) for y in x] for x in output]
                    res_sents = [[tokenizer.convert_ids_to_tokens(y) for y in x] for x in res_sents_ids]
                    vocab_len = [processor.get_vocab_length("".join(tokens)) for tokens in res_sents]
                    vocab_mean = np.mean(vocab_len)
                    vocab_len = [(x - vocab_mean) for x in vocab_len]
                    scores = [weight * v_l + sum(score) for v_l, score in zip(vocab_len, res_sents_scores)]
                    # print(scores)
                    for idx, (_, s) in enumerate(zip(mid_results, scores)):
                        mid_results[idx][-1] = s
                    # sort results by scores
                    mid_results = sorted(mid_results, key=lambda x: -x[-1])[:beam_size]

            res_sents = convert_type(mid_results)
            return res_sents
        res_sents = beam_search(res_sent)


        res_sents_ids = [[int(y.split(",")[1]) for y in x] for x in res_sents]
        res_sents_scores = [[float(y.split(",")[0]) for y in x] for x in res_sents]

        res_sents = [[tokenizer.convert_ids_to_tokens(y) for y in x] for x in res_sents_ids]
        vocab_len = [processor.get_vocab_length("".join(tokens)) for tokens in res_sents]
        vocab_mean = np.mean(vocab_len)
        vocab_len = [(x - vocab_mean) for x in vocab_len]
        scores = [weight * v_l + sum(score) for v_l, score in zip(vocab_len, res_sents_scores)]

        largest_prob = 0
        for k in range(len(scores)):
            if scores[largest_prob] < scores[k]:
                largest_prob = k
        res_sent_id = res_sents_ids[largest_prob]
        res_sent = res_sents[largest_prob]


        return res_sent_id, res_sent
 
    
    for input in tqdm(data):
            i += 1
            src_ids = tokenizer(input.src,
                                max_length = args.max_seq_length // 2 - 2,
                                truncation=True,
                                is_split_into_words=True,
                                add_special_tokens=False).input_ids
            flag = False
            for chids in src_ids:
                if chids == tokenizer.unk_token_id:
                    flag=True
                    break
            if flag == True:
                continue
            trg_ids = tokenizer(input.trg,
                                max_length = args.max_seq_length // 2 - 2,
                                truncation=True,
                                is_split_into_words=True,
                                add_special_tokens=False).input_ids
            
            src_ids = [tokenizer.cls_token_id] + src_ids + [tokenizer.sep_token_id] + [tokenizer.mask_token_id for _ in src_ids] + [tokenizer.sep_token_id]
            trg_ids = [tokenizer.cls_token_id] + trg_ids + [tokenizer.sep_token_id] + trg_ids + [tokenizer.sep_token_id] 
            attention_mask = [1] * len(src_ids)
            src_ids = torch.LongTensor([src_ids]).to("cuda")
            attention_mask = torch.LongTensor([attention_mask]).to("cuda")
            trg_ids = torch.LongTensor([trg_ids]).to("cuda")    
            outputs = model(src_ids)
            logits = outputs["logits"]
            probs = model.softmax(logits)
            prd_probs, prd_ids = torch.max(probs, -1)
            _p = [pt for pt, st in zip(prd_ids[0], src_ids[0]) if st == tokenizer.mask_token_id]
            _pp = [pt for pt, st in zip(prd_probs[0], src_ids[0]) if st == tokenizer.mask_token_id]
            
            inp = decode(src_ids[0])
            # print("INPUT:",inp)
            # exit()
            all_inputs += [inp]
            _t = [tt for tt,st, in zip(trg_ids[0],src_ids[0]) if st == tokenizer.mask_token_id]
            # print((stdf(" ".join(input.src))).split())
            all_trgs += [decode(_t)]
            # print("TRG:",decode(_t))
            # print(decode(_p))
            input_ids = _p
            xx = _pp
            if args.calc_prd:
                if args.pick_max:
                    corr = 0
                    for _ in range(len(input_ids)):
                        max_prob = 0
                        letter_ids = -1
                        idx = 0
                        # min_prob = min(xx[j],prd_probs[0][j+1])
                        for j in range(len(input_ids)):        
                            if round(xx[j].item(), 4)==1 and round(prd_probs[0][j+1].item(),4)==1:
                                continue
                            tx = input_ids[:]
                            ori = tx[j]
                            tx[j] = tokenizer.mask_token_id
                            _p,_pp = calc_prd(tx,True,ori,j)
                            if _p[j] != tokenizer.unk_token_id and _p[j] != ori and _pp[j] > max_prob:
                                letter_ids = _p[j]
                                idx = j
                                max_prob = _pp[j]
                        # print(max_prob)
                        if letter_ids != -1 and max_prob > args.change_logits:
                            input_ids[idx] = letter_ids
                            input_ids,xx = calc_prd(input_ids,False)
                            corr = corr + 1
                            if corr == 2:
                                break
                        # elif letter_ids != -1 and max_prob > 0.3:
                        #     tx[j] = _p[j]
                        #     p,pp = calc_prd(tx,False)
                        #     if pp[j] > min_prob:
                        #         input_ids[idx] = letter_ids
                        #         input_ids,xx = calc_prd(input_ids,False)
                        #         src_ids = [tokenizer.cls_token_id] + src_ids + [tokenizer.sep_token_id] + [tokenizer.mask_token_id for _ in src_ids] + [tokenizer.sep_token_id]
                        #         src_ids = torch.LongTensor([src_ids]).to("cuda")
                        #         outputs = model(src_ids)
                        #         logits = outputs["logits"]
                        #         probs = model.softmax(logits)
                        #         prd_probs, prd_ids = torch.max(probs, -1)
                        else: 
                            # print(_)
                            break
                elif args.accelerate: 
                    # sort_idx = sorted(range(len(xx)),key=lambda k: xx[k], reverse=False)
                    # print(xx[sort_idx[0]])
                    # print(xx[sort_idx[1]])
                    # print(sort_idx)
                    # exit()
                    # for _ in range(len(input_ids)):
                        max_prob = 0
                        letter_ids = -1
                        idx = 0
                        sort_idx = sorted(range(len(xx)),key=lambda k: xx[k], reverse=False)
                        for idx_ in range(args.acc_range):
                            if idx_ == len(sort_idx): break
                            r = sort_idx[idx_]
                            
                            # for j in range(len(input_ids)):
                            # min_prob = min(xx[r],prd_probs[0][r+1])
                            tx = input_ids[:]
                            ori = tx[r]
                            tx[r] = tokenizer.mask_token_id
                            _p,_pp = calc_prd(tx,True,ori,r)
                            if _p[r] != tokenizer.unk_token_id and _p[r] != ori and _pp[r] > max_prob:
                                letter_ids = _p[r]
                                idx = r
                                max_prob = _pp[r]
                            if letter_ids != -1 and max_prob > args.change_logits:
                                max_prob=0
                                input_ids[idx] = letter_ids
                                input_ids,xx = calc_prd(input_ids,False)
                            else: break
                        # else: 
                        #     break
                            # elif letter_ids != -1:
                            #     tx[r] = _p[r]
                            #     p,pp = calc_prd(tx,False)
                            #     if pp[r] > min_prob:
                            #         input_ids[idx] = letter_ids
                            #         input_ids,xx = calc_prd(input_ids,False) 

                else:
                    for _ in range(len(input_ids)):
                        max_prob = 0
                        letter_ids = -1
                        idx = 0
                        # for j in range(len(input_ids)):
                        min_prob = min(xx[_],prd_probs[0][_+1])
                        if round(xx[_].item(), 4)==1 and round(prd_probs[0][_+1].item(),4)==1:
                            continue
                        tx = input_ids[:]
                        ori = tx[_]
                        tx[_] = tokenizer.mask_token_id
                        _p,_pp = calc_prd(tx,True,ori,_)
                        if _p[_] != tokenizer.unk_token_id and _p[_] != ori and _pp[_] > max_prob:
                            letter_ids = _p[_]
                            idx = _
                            max_prob = _pp[_]
                        if letter_ids != -1 and max_prob > args.change_logits:
                            input_ids[idx] = letter_ids
                            input_ids,xx = calc_prd(input_ids,False)
                        elif letter_ids != -1:
                            tx[_] = _p[_]
                            p,pp = calc_prd(tx,False)
                            if pp[_] > min_prob:
                                input_ids[idx] = letter_ids
                                input_ids,xx = calc_prd(input_ids,False) 

                if args.use_ud:
                    k_size = 5
                    top_k_scores, log_topk_scores, topk_indexes = calc_topk(input_ids,k_size) 
                    input_ids, prd_chr = customize(top_k_scores,log_topk_scores, topk_indexes,processor,tokenizer,k_size,len(input_ids))

            
            
                prd = (stdf(" ".join(decode(input_ids)))).split()
                prd = [p if p != "[UNK]" else s for p,s in zip(prd,inp)] 
                # print("PRD:",prd)
                all_prd += [prd]
            


            else:
                if args.use_ud:
                    k_size = 5
                    top_k_scores, log_topk_scores, topk_indexes = calc_topk(input_ids,k_size) 
                    input_ids, prd_chr = customize(top_k_scores,log_topk_scores, topk_indexes,processor,tokenizer,k_size,len(input_ids))    
                
                prd = (stdf(" ".join(decode(input_ids)))).split()
                prd = [p if p != "[UNK]" else s for p,s in zip(prd,inp)] 
                all_prd += [prd]

            if i == 10:
                output_debug = os.path.join(args.output_dir,"all.debug")
                for ss,tt,pp in zip(all_inputs,all_trgs,all_prd):
                    with open(output_debug,"a") as writer:
                        writer.write("S: " + " ".join(ss) + "\n")
                        writer.write("T: " + " ".join(tt) + "\n")
                        writer.write("P: " + " ".join(pp) + "\n")
                # exit()
            if i % 100 == 0:
                output_result_file = os.path.join(args.output_dir,"sents.result")
                p,r,f1,fpr,tp,fp,fn = Metrics.compute(all_inputs,all_trgs,all_prd)
                print(f1*100)
                # print(tp)
                # print(fp)
                # print(fn)
                with open(output_result_file, "a") as writer: 
                    writer.write(str(i) + " steps:" + "\n")
                    writer.write("F1:" + str(f1 * 100) + "\n")
                    writer.write("P:" + str(p * 100) + "\n")
                    writer.write("R:" + str(r * 100) + "\n")
                    writer.write("FPR:" + str(fpr * 100) + "\n")
               

            


    p,r,f1,fpr,tp,fp,fn = Metrics.compute(all_inputs,all_trgs,all_prd)
    endt = datetime.datetime.now()
    with open(output_result_file, "a") as writer: 
                    writer.write(str(i) + " steps:" + "\n")
                    writer.write("F1:" + str(f1 * 100) + "\n")
                    writer.write("P:" + str(p * 100) + "\n")
                    writer.write("R:" + str(r * 100) + "\n")
                    writer.write("FPR:" + str(fpr * 100) + "\n")
    print(f1*100)
    
    output_tp_file = os.path.join(args.output_dir, "sents.tp")
    print(output_tp_file)
    with open(output_tp_file, "w") as writer:
        for line in tp:
            writer.write(line + "\n")
        print((endt - startt).seconds)
        writer.write(str((endt - startt).seconds))
        
    output_fp_file = os.path.join(args.output_dir, "sents.fp")
    with open(output_fp_file, "w") as writer:
        for line in fp:
            # print(line)
            writer.write(line + "\n")
    output_fn_file = os.path.join(args.output_dir, "sents.fn")
    with open(output_fn_file, "w") as writer:
        for line in fn:
            writer.write(line + "\n")
    
if __name__ == "__main__":
    main()