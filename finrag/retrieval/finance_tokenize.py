# -*- coding: utf-8 -*-
"""finance_tokenize.py — 带金融词典的 jieba 分词（索引与查询共用，保证一致）"""
import os
import jieba

jieba.setLogLevel(20)
_DICT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "finance_dict.txt")
if os.path.exists(_DICT):
    jieba.load_userdict(_DICT)


def tokenize(text):
    return jieba.lcut(text or "")
