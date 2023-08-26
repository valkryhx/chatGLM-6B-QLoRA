# 构造带上下文的alpaca格式数据



import json

list_dict =[]
with open("yunguan_aug_part_1.json","r",encoding="utf-8") as fr:
    list_dict .extend([json.loads(item) for item in fr.readlines()])

with open("yunguan_aug_part_2.json","r",encoding="utf-8") as fr:
    list_dict .extend([json.loads(item) for item in fr.readlines()])
    
with open("yunguan_aug_part_3.json","r",encoding="utf-8") as fr:
    list_dict.extend([json.loads(item) for item in fr.readlines()])

with open("refuse_phone.json","r",encoding="utf-8") as fr:
    list_dict.extend([json.loads(item) for item in fr.readlines()])
print(len(list_dict))

prompt = "现在提供如下信息：\n{}\n{}\n{}\n请根据提供的信息回答问题。问：{}"

res =[]
import random
for idx,item in enumerate(list_dict):
    question = item["instruction"]
    response_j = item["output"]
    # make bad_answer response_k
    is_duplicate = True
    two_bad_answers = []
    list_context =[]
    
    while is_duplicate == True:
        bad_idx = random.choice(range(len(list_dict)))
        #bad_idx = bad_idx  if bad_idx !=idx else (idx+1)%len(list_dict)
        #assert bad_idx != idx , "bad_idx should not be equal to idx"
        #print(idx,bad_idx)
        response_k = list_dict[bad_idx]["output"]
        if response_j != response_k:
            two_bad_answers.append(response_k) # 坏回答存储起来 后面从中选一个作为response_k
            list_context.append(list_dict[bad_idx]['instruction']+list_dict[bad_idx]['output'])
        if len(two_bad_answers) == 2:
            # 累积 2个随机上下文才退出
            is_duplicate = False
    # 累积两个随机无关上下文后 把正确的上下文也加入 然后shuffle一下三个上下文的顺序
    list_context.append(question+response_j)
    random.shuffle(list_context)
    
    question_with_context = prompt.format(*list_context,question)
    bad_answer_from_context = random.choice(two_bad_answers)
    res.append(
        {
            "question" : question_with_context ,
            "response_j" : response_j ,
            "response_k" : bad_answer_from_context,
        }
    )

#shuffle res
random.shuffle(res)
print(len(res))
with open("paired_yunguan_with_2_bad_context_1_righ_tcontext.json","w",encoding="utf-8") as fw :
    for paired_item in res :
        fw.write(json.dumps(paired_item,ensure_ascii=False) +"\n")
print("write done")


# 查看一下sample各个字段的长度

max_prompt,max_response_j,max_response_k,max_sample =0,0,0,0
for item in res[:453] :
    max_question = max(len(item['question']),max_prompt)
    max_response_j= max(len(item['response_j']),max_response_j)
    max_response_k= max(len(item['response_k']),max_response_k)
    max_sample = max(len(item['question']) + len(item['response_j']) + len(item['response_k'])   , max_sample)
print(f"len_max_prompt={max_question}\n \
len_max_response_j={max_response_j}\n \
len_max_response_k={max_response_k}\n \
len_max_sample={max_sample}")
