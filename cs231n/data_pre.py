# -*- coding: utf-8 -*-
from snownlp import SnowNLP
import re
import HTMLParser
h = HTMLParser.HTMLParser()


def filter_other(str):    
    re_symbol = re.compile('[\[\]\"\'><\(\){}]')
    s = re_symbol.sub(' ', str)
    re_chinese_symbol=re.compile(u'[\uFF08\uFF09\u3014\u3015\u3010\u3011\u2026\u300A\u300B\u2014\u3008\u3009\u201C\u201D\u2018\u2019]')
    #FF08:（ FF09:） 3014:( 3015:) 3010:【 3011:】 2026:... 300A:《 300B:》 2014:— 3008:< 3009:> 201C:“ 201D:” 2018:‘ 2019:’    
    s = re_chinese_symbol.sub(' ', s)
    return s

def isolate_symbol(str):
    re_symbol = re.compile(',')
    s = re_symbol.sub(' , ', str)
    re_symbol = re.compile(':')
    s = re_symbol.sub(' : ', s)
    #re_symbol = re.compile('!')
    #s = re_symbol.sub(' ! ', s)
    re_symbol = re.compile('\?')
    s = re_symbol.sub(' ? ', s)
    re_chinese_symbol=re.compile(u'\u3001') #、
    s = re_chinese_symbol.sub(u' \u3001 ', s)
    re_chinese_symbol=re.compile(u'\uFF0C') #,
    s = re_chinese_symbol.sub(u' \uFF0C ', s)
    re_chinese_symbol=re.compile(u'\uFF1A') #:
    s = re_chinese_symbol.sub(u' \uFF1A ', s)
    #re_chinese_symbol=re.compile('、') #、
    #s = re_chinese_symbol.sub(' 、 ', s)
    #re_chinese_symbol=re.compile('，') #,
    #s = re_chinese_symbol.sub(' ， ', s)
    #re_chinese_symbol=re.compile('：') #,
    #s = re_chinese_symbol.sub(' ： ', s)
    
    return s


def en_pre(str):
    str=str.strip().rstrip('.')          
    str=h.unescape(str.decode('utf-8'))        
    str=str.lower()
    str=filter_other(str)
    str=isolate_symbol(str)  
    str_list=str.strip().split()
    return str_list


def cn_pre(str):
    str=str.decode('utf-8').lower().strip().rstrip('.')
    str=h.unescape(str)
    re_symbol = re.compile(' ')
    str = re_symbol.sub('', str)
    
    str=filter_other(str)
    str_list=SnowNLP(str).words
    return str_list



