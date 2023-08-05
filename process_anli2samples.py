#!/usr/bin/env python
# coding: utf-8

# In[41]:


from glob import glob
def get_all_files(directory, extension):
    """Get all files with a specific extension in a directory and its subdirectories using glob"""
    return glob(f'{directory}/**/*.{extension}', recursive=True)


# In[42]:


#get_all_files(directory="json示例/" , extension = "txt")


# In[43]:


import codecs
import json
def read_txt_utf8_bom(filename):
    with codecs.open(filename, 'r', 'utf-8-sig') as fr:
        content = json.load(fr)
    return content


# In[44]:


#a=read_txt_utf8_bom(get_all_files(directory="json示例/" , extension = "txt")[0])


# In[45]:


import re

def remove_html_tags(text):
    """Use regex to remove all html tags from a string"""
    clean = re.compile('<.*?>')
    nbsp = re.compile("&nbsp;")
    _r = re.compile("\r\n")
    text = re.sub(clean, '', text)
    text = re.sub(nbsp,"",text)
    text = re.sub(_r,"",text)
    return text


# In[46]:


#b=remove_html_tags(a['0']["clgc"])


# In[47]:


desc={
    "ankzt":"案例主题",
    "name":"案例编号",
    "wlcj":"网络场景",
    "xxms":"现象描述",
    "clgc":"处理过程",
    "gy":"根因",
    "jjfa":"解决方案",
    "yyfl":"原因分类"
}


# In[66]:


def fill_null(a):
    if a is None:
        return "无"
    return a


if __name__ == '__main__':
    import argparse
    def parse_args():
        parser = argparse.ArgumentParser(description='预处理案例库文件')
        parser.add_argument('--in_files_path', type=str, required=True, help='存放案例库的目录位置')
        parser.add_argument('--file_type', type=str,default="txt", required=True, help='案例文件后缀，例如txt')
        parser.add_argument('--output_file', type=str,default="out_anli.json", required=True, help='处理完后存放样本的文件')
        return parser.parse_args()

    args = parse_args()

    res=[]
    max_inst_len = -1
    max_inp_len =-1
    max_out_len =-1
    PROMPT = "作为烽火智能助手，我遇到了以下故障现象，{}。请给出类似案例以供参考。"
    case_file_list = get_all_files(directory=args.in_files_path , extension = args.file_type)
    print(case_file_list)
    for case_name in case_file_list :
        print(f"正在处理:{case_name}")
        d = read_txt_utf8_bom(filename=case_name)['0'] # get dict of 案例 ,注意取d['0']对应的内层dict
    #     xxms = remove_html_tags(d.get('xxms',"无"))
    #     ankzt = remove_html_tags(d.get('ankzt',"无"))
    #     name = remove_html_tags(d.get('name',"无"))
    #     wlcj = remove_html_tags(d.get('wlcj',"无"))
    #     clgc = remove_html_tags(d.get('clgc',"无"))
    #     gy = remove_html_tags(d.get('gy',"无"))
    #     jjfa = remove_html_tags(d.get('jjfa',"无"))
    #     yyfl = remove_html_tags(d.get('yyfl',"无"))
    
        instruction = PROMPT.format(remove_html_tags(d.get("xxms","无")))
        input_=""
        keys = ['ankzt','name','wlcj','xxms','clgc','gy','jjfa','yyfl']
        for k in keys :
            print(f"key={k}")
            print(remove_html_tags(fill_null(d.get(k,"无"))))
        output = "".join( [desc[k]+":" + remove_html_tags(fill_null(d.get(k,"无"))) + "\n" for k in keys ] ) 
        res.append({"instruction":instruction ,"input":input_ , "output":output})
        max_inst_len = max(max_inst_len,len(instruction))
        max_inp_len = max(max_inp_len,len(input_))
        max_out_len = max(max_out_len,len(output))
        print(f"处理完毕:{case_name}")
    print(res)
    print(f"\n共收集样本数量:{len(res)}")
    print(f"\n\nmax_instruction_length={max_inst_len}\nmax_input_length={max_inp_len}\nmax_output_length={max_out_len}")


# In[76]:


    with open(args.output_file,"w",encoding="utf-8" ) as fw:
        for item in res :
            fw.write(json.dumps(item,ensure_ascii=False)+"\n")
    print(f"样本json文件为：{args.output_file} write done")


# In[78]:



#args = parse_args()
#args.dir args.file_type,args.output_file


# In[ ]:

"""
    usage:
    python process_anli2samples.py  --in_files_path "案例库-正文" --file_type txt --output_file  anli_0805.json
"""




