# MIT License
# 
# Copyright (c) 2015-2025 Aigents®, Anton Kolonin 
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from pygents.text import get_grams
from pygents.util import cosine_similarity, dictcount
from pygents.aigents_api import tokenize_re, punct

def get_char_grams(items,n=2):
    grams = []
    for i in range(len(items) - (n-1)):
        gram = None
        for j in range(n):
            gram = items[i+j] if gram is None else gram + items[i+j]
        grams.append(gram)
    return grams

def get_char_grams_dicts(items,n=2):
    grams = get_char_grams(items,n)
    d = {}
    for g in grams:
        dictcount(d,g)
    return d

def texts_char_grams_counts(texts,n=2):
    grams = {}
    for t in texts:
        grams[t] = get_char_grams_dicts(t.lower(),2)
    return grams

assert(str(get_char_grams_dicts("Hello world, hello"))=="{'He': 1, 'el': 2, 'll': 2, 'lo': 2, 'o ': 1, ' w': 1, 'wo': 1, 'or': 1, 'rl': 1, 'ld': 1, 'd,': 1, ', ': 1, ' h': 1, 'he': 1}")
assert(str(texts_char_grams_counts(["ping","pong"]))=="{'ping': {'pi': 1, 'in': 1, 'ng': 1}, 'pong': {'po': 1, 'on': 1, 'ng': 1}}")

def get_item_grams(items,n=2):
    grams = []
    for i in range(len(items) - (n-1)):
        gram = []
        for j in range(n):
            gram.append(items[i+j])
        grams.append(tuple(gram))
    return grams

def get_item_grams_dicts(items,n=2):
    grams = get_item_grams(items,n)
    d = {}
    for g in grams:
        dictcount(d,g)
    return d

def texts_item_grams_counts(texts,n=2,clean_punct=False):
    grams = {}
    for text in texts:
        tokens = tokenize_re(text.lower())
        if clean_punct:
            tokens = [t for t in tokens if not (t in punct or t.isnumeric())]
        grams[text] = get_item_grams_dicts(tokens,2)
    return grams

assert(str(get_item_grams_dicts(tokenize_re("Hello world, hello world")))=="{('hello', 'world'): 2, ('world', ','): 1, (',', 'hello'): 1}")
assert(str(texts_item_grams_counts(["play ping pong!","sing ding dong..."]))=="{'play ping pong!': {('play', 'ping'): 1, ('ping', 'pong'): 1, ('pong', '!'): 1}, 'sing ding dong...': {('sing', 'ding'): 1, ('ding', 'dong'): 1, ('dong', '.'): 1, ('.', '.'): 2}}")
assert(str(texts_item_grams_counts(["play ping pong!","sing ding dong..."],clean_punct=True))=="{'play ping pong!': {('play', 'ping'): 1, ('ping', 'pong'): 1}, 'sing ding dong...': {('sing', 'ding'): 1, ('ding', 'dong'): 1}}")

class FuzzyMatcher:

    def __init__(self, texts, options=('wordsonly','wordschars','chars')):
        self.idxs = {}
        self.idxs['chars'] = texts_char_grams_counts(texts) if 'chars' in options else None 
        self.idxs['wordschars'] = texts_item_grams_counts(texts) if 'wordschars' in options else None
        self.idxs['wordsonly'] = texts_item_grams_counts(texts,clean_punct=True) if 'wordsonly' in options else None
        #print(self.idxs)
        #print()
            
    def match(self,text,options=('wordsonly','wordschars','chars'),threshold = 0.3,debug = False):
        for option in options:
            if debug:
                print(option)
            if option in self.idxs:
                idx = self.idxs[option]
                if debug:
                    print(idx)
                if idx is None: # not initialized
                    continue
                if option == 'wordsonly':
                    tokens = tokenize_re(text.lower())
                    tokens = [t for t in tokens if not (t in punct or t.isnumeric())]
                    sample = get_item_grams_dicts(tokens,2)
                elif option == 'wordchars':
                    tokens = tokenize_re(text.lower())
                    sample = get_item_grams_dicts(tokens,2)
                elif option == 'chars':
                    sample = get_char_grams_dicts(text.lower(),2)
                if debug:
                    print(sample)
                maxsim = 0
                bestmatch = None
                for i in idx:
                    #print(idx[i])
                    sim = cosine_similarity(idx[i],sample)
                    if sim >= threshold:
                        if maxsim < sim:
                            maxsim = sim
                            bestmatch = i
                if not bestmatch is None:
                    return bestmatch, maxsim, option
        return None, None, None

    def auto_correct_tokens(self,tokens,threshold=0.8):
        new_list = []
        lex = self.idxs['chars']
        for t in tokens:
            if t in lex:
                new_list.append(t)
            else:
                bestmatch, maxsim, option = self.match(t,threshold=threshold)
                new_list.append(t if bestmatch is None else bestmatch)
        return new_list if type(tokens) == list else tuple(new_list)

fm = FuzzyMatcher(['Anton Kolonin','Evgeny Bochkov','Alexey Gluschshenko','International Business Machines'])

assert(str(fm.match('Alexey'))=="('Alexey Gluschshenko', 0.53, 'chars')")
assert(str(fm.match('Alex'))=="('Alexey Gluschshenko', 0.41, 'chars')")
assert(str(fm.match('International Business'))=="('International Business Machines', 0.71, 'wordsonly')")
assert(str(fm.match('Interational Busines'))=="('International Business Machines', 0.84, 'chars')")
