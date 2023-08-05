import json
import argparse


def sharegpt_to_history(f_in,f_out):
    fw = open(f_out,"w",encoding="utf-8")
    valid_n =0
    original_n = 0
    with open(f_in,"r",encoding="utf-8") as fr:
        
        for item in fr.readlines():
            conv = json.loads(item)["conversations"]
            original_n += 1
            #if len(conv)%2 !=0 :
            #    print("cut off to 2n," ,len(conv),conv)
            valid_len = len(conv)//2*2
            if valid_len ==0:
                print("***only 1 str ,not valid, will discard***")
                print(conv)
                continue
            history =[]
            for idx in range(0,valid_len,2) :  #一行的sharegpt sample
                #print(idx)
                if conv[idx] and  conv[idx].get("from","")== "human" and   conv[idx+1] and  conv[idx+1].get("from","")== "gpt":
                    one_turn = [conv[idx].get("value")   , conv[idx+1].get("value") ] 
                    history.append(one_turn) # 将一行sharegpt sample的每个turn 存入一个history格式的sample中
            fw.write(json.dumps({"history":history},ensure_ascii=False)+"\n")
            valid_n += 1
    fw.close()
    print("done")
    print(f"原始对话有{original_n}")
    print(f"转化成功对话{valid_n}")
    
if __name__=="__main__":
    parser = argparse.ArgumentParser(description='shareGPT_to_history')
    parser.add_argument('--file_in', type=str, required=True, help='sharegpt format input json / jsonl file')
    parser.add_argument('--file_out', type=str, required=True, help='history format output json / jsonl file')
    args = parser.parse_args()
    sharegpt_to_history(f_in=args.file_in,f_out=args.file_out)
    
    """
    usage :  python shareGPT_to_history.py --file_in sharegpt_zh_1K.json --file_out h_1000.json
    """