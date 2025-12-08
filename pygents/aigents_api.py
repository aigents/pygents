# MIT License
# 
# Copyright (c) 2015-2025 Aigents®, Anton Kolonin, Anna Arinicheva
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

from _ast import arg

import os
import requests
import urllib.parse
import math
import json
import re
import emoji
from collections import defaultdict

#https://towardsdatascience.com/5-simple-ways-to-tokenize-text-in-python-92c6804edfc4
#https://www.oreilly.com/content/how-can-i-tokenize-a-sentence-with-python/
import nltk

#TODO run this once to make it working
nltk.download('punkt')
SENT_DETECTOR = nltk.data.load('tokenizers/punkt/english.pickle')

from pygents.text import url_lines
from pygents.util import dictcount, dictdict_div_dict, dict_of_dicts_compress_by_threshold, dictdict_mul_dictdict

import logging
logger = logging.getLogger(__name__)    

"""
Service wrapper around Aigents Java-based Web Service
"""        
class AigentsAPI:

    def __init__(self, base_url='https://aigents.com/al', login_email='john@doe.org', secret_question='password', secret_answer='1234567', real_mode=True, name='Aigents', verbose=False):
        self.name = name
        self.verbose = verbose
        self.base_url = base_url # Aigents Web API hosting URL
        self.login_email = login_email # Aigents user identification by email 
        self.secret_question = secret_question # Aigents prompt for password
        self.secret_answer = secret_answer # Aigents password value
        self.real_mode = real_mode # whether to connect to real Aigents server (True) or fake test oe (False) 
        if self.verbose:
            logger.info('Creating Aigents session')
        #print(self)
        self.create_session()
    
    def __del__(self):
        if self.verbose:
            logger.info('Closing Aigents session')
        self.close_session()

    def create_session(self):
        self.session = requests.session()
        if self.real_mode:
            #TODO assertions
            self.request('my email ' + self.login_email + '.')
            self.request('my ' + self.secret_question + ' '  + self.secret_answer + '.')
            self.request('my language english.')
        else:
            #TODO make sure if we can use only one of these
            output = self.request('my name ' + self.login_email + ', surname ' + self.login_email + ', email ' + self.login_email + '.')
            if output == 'What your secret question, secret answer?':
                assert output == 'What your secret question, secret answer?', 'Expecting secret question, secret answer'
                output = self.request('my secret question ' + self.secret_question + ', secret answer ' + self.secret_answer + '.')
            assert output == 'What your ' + self.secret_question + '?', 'Expecting secret question'
            output = self.request('my ' + self.secret_question + ' ' + self.secret_answer + '.')
            assert output.split()[0] == 'Ok.', 'Expecting Ok'

    def close_session(self):
        if not self.real_mode:
            output = self.request('Your trusts no ' + self.login_email + '.')
            assert output.split()[0] == 'Ok.', 'Expecting Ok'
            output = self.request('No name ' + self.login_email + '.');
            assert output.split()[0] == 'Ok.', 'Expecting Ok'
            output = self.request('No there times today.');
            assert output.split()[0] == 'Ok.', 'Expecting Ok'
        output = self.request('My logout.');
        assert output.split()[0] == 'Ok.', 'Expecting Ok'
            
    def request(self,input):
        if self.verbose:
            logger.info(input)
        url = self.base_url + '?' + urllib.parse.quote_plus(input)
        try:
            r = self.session.post(url)
            if r is None or r.status_code != 200:
                logger.error('request ' + url + ' error ' + str(r.status_code))
                raise RuntimeError("Aigents - no response")
        except Exception as e:
            logger.error('request ' + url + ' ' + str(type(e)))
            print('Specify proper url to Aigents server or run it locally, eg.')
            print('java -cp ./bin/Aigents.jar:./bin/* net.webstructor.agent.Farm store path \'./al_test.txt\', cookie domain localtest.com, console off &')
            print('or')
            print('sh aigents_server_start.sh')
            return 'No connection to Aigents, ' + str(type(e))
        if self.verbose:
            logger.info(r.text)
        return r.text
    
    
class AigentsSentiment():

    def __init__(self, api_url='http://localtest.com:1180/', debug=False):
        self.aa = AigentsAPI(api_url)
        self.debug = debug
        reply = self.aa.request("my format json")
        if debug:
            print(reply)
            
    def get_sentiment(self,text,context=None,debug=False):
        
        wordcnt = len(text.strip().split(' '))
        itemcnt = 1
    
        text_quoted = '"'+text.replace('"','\\"')+'"'
        request_text = "classify sentiment text "+text_quoted
        json_text = self.aa.request(request_text)
        if self.debug:
            print('--------')
            print('text:',text) 
            print('json:',json_text)
            
        sen = 0
        pos = 0
        neg = 0
        con = 0
            
        try:
            python_data = json.loads(json_text)
            item = python_data[0]
            #print(text, item['sentiment'])
            sen = float(item['sentiment']) / 100
            pos = float(item['positive']) / 100 if 'positive' in item.keys() else 0
            neg = -float(item['negative']) / 100 if 'negative' in item.keys() else 0
            con = round(math.sqrt(pos * -neg),2)
        except Exception as e:
            print(str(e), request_text,'=>',json_text)
            if self.debug:
                raise e
        return sen, pos, neg, con, wordcnt, itemcnt
    

def token_valid(token):
    if len(token) < 1:
        return False
    if re.match(r'^\(?(http|https)://',token) is not None:
        return False
    return True

soft_delimiters_regexp = r' |\n|\t|\r|\[|\]|\(|\)'
hard_delimiters_list = ",;:.!?"
apostrophes_list = "'‘’"
quotes_list = "`\"“”„(){}[]<>"
inner_word_punctuation_list = "-"
diverse_punctuation_list = "_…–&•—$+\\/*=#^@~‰"
any_punctuation_list = inner_word_punctuation_list + diverse_punctuation_list 
word_splitters = hard_delimiters_list + diverse_punctuation_list + quotes_list
punct = any_punctuation_list + quotes_list + hard_delimiters_list # for external use and scrub filtering

def add_token(token,res_list):
    if len(token) == 0:
        return
    if len(token) == 1:
        res_list.append(token)
        return
    first = token[0]
    last = token[-1]
    if first in emoji.UNICODE_EMOJI['en'] or first in hard_delimiters_list or first in any_punctuation_list or first in quotes_list or first in apostrophes_list:
        res_list.append(first)
        add_token(token[1:],res_list)
        return
    if last in emoji.UNICODE_EMOJI['en'] or last in hard_delimiters_list or last in any_punctuation_list or last in quotes_list or last in apostrophes_list:
        add_token(token[0:-1],res_list)
        res_list.append(last)
        return
    if token_valid(token): # and len(token) > 1 !!!
        #if len(token) > 2 and token[0].isalpha() and token[-1].isalpha():
        if len(token) > 2 and token[0].isalpha(): # pretend its a word!
            for delimiter in word_splitters:
                if token.find(delimiter) != -1:
                    tokens = token.split(delimiter)
                    for i in range(0,len(tokens)):
                        if i > 0:
                            res_list.append(delimiter)
                        add_token(tokens[i],res_list)
                    return
        res_list.append(token)

def tokenize_re(text):
    text = text.replace(u'\xa0', u' ')
    tokens = re.split(soft_delimiters_regexp,text.lower())
    res_list = []
    for token in tokens:
        if len(token) < 1:
            continue
        add_token(token,res_list)
    return res_list
assert(str(tokenize_re('I like ‘me’ ’cause don’t like ‘it’.')) == "['i', 'like', '‘', 'me', '’', '’', 'cause', 'don’t', 'like', '‘', 'it', '’', '.']")
assert(str(tokenize_re('My know-how, #1?')) == "['my', 'know-how', ',', '#', '1', '?']")
assert(str(tokenize_re('My 25 -30 partners (at all)!')) == "['my', '25', '-', '30', 'partners', 'at', 'all', '!']")
assert(str(tokenize_re('of …it’s my fault ….i should die … like ..i worked')) == "['of', '…', 'it’s', 'my', 'fault', '…', '.', 'i', 'should', 'die', '…', 'like', '.', '.', 'i', 'worked']")
assert(str(tokenize_re('him „faithful‰ to')) == "['him', '„', 'faithful', '‰', 'to']")
assert(str(tokenize_re('I.like…tea')) == "['i', '.', 'like', '…', 'tea']")

def build_ngrams(seq,N):
    size = len(seq) - N + 1;
    if size < 1:
        return [];
    items = [];
    for i in range(0,size):
        items.append( tuple(seq[i:i+N]) )
    return items

def load_ngrams(file,encoding=None,weights=None,debug=False):
    if file.lower().startswith('http'):
        lines = url_lines(file)
        if debug:
            print(lines[:100])
            print(len(lines))
    else:
        if encoding is None:
            with open(file) as f:
                lines = f.readlines()
        else:
            with open(file,encoding=encoding) as f:
                lines = f.readlines()
    if not weights is None:
        ngrams = []
        for l in lines:
            if len(l) > 0:
                split = l.split('\t')
                ngram = tuple(split[0].split())
                weight = float(split[1]) if len(split) > 0 else 1.0
                weights[ngram] = weight
                ngrams.append(ngram)
    else:
        ngrams = [tuple(l.split('\t')[0].split()) for l in lines if len(l) > 0]
    return set(ngrams)

def split_pattern(pat):
    new_list = []
    for p in pat:
        new_list.extend([*p])
    return tuple(new_list)
assert str(split_pattern(('ab','c','d'))) == "('a', 'b', 'c', 'd')"

def split_patterns(arg):
    return [split_pattern(a) for a in arg]
assert str(split_patterns([('abcd',),('ef','gh'),('i','j')])) == "[('a', 'b', 'c', 'd'), ('e', 'f', 'g', 'h'), ('i', 'j')]"


#TODO move to data ("model") files
scrub_en = ["-", "&","a", "an", "and", "because", "else", "or", "the", "in", "on", "at", "it", "is", "after", "are", "me",
                    "am", "i", "into", "its", "same", "with", "if", "most", "so", "thus", "hence", "how",
                    "as", "do", "what", "for", "to", "of", "over", "be", "will", "was", "were", "here", "there",
                    "you", "your", "our", "my", "her", "his", "just", "have", "but", "not", "that",
                    "their", "we", "by", "all", "any", "anything", "some", "something", "dont", "do", "does", "of", "they", "them",
                    "been", "even", "etc", "this", "that", "those", "these", "from", "he", "she",
                    "no", "yes", "own", "may", "mine", "me", "each", "can", "could", "would", "should", "since", "had", "has",
                    "when", "out", "also", "only", "about", "us", "via", "than", "then", "up", "who", "why", "which", "yet"]

class PygentsSentiment():

    def __init__(self, positive_lexicon_file, negative_lexicon_file, sentiment_maximized=False, sentiment_logarithmic=True, tokenize_chars=False, encoding=None, debug=False):
        self.sentiment_maximized = sentiment_maximized
        self.sentiment_logarithmic = sentiment_logarithmic
        self.gram_arity = 3
        self.scrub = set(scrub_en)
        self.punct = set(list(punct))
        self.positives = self.to_set(positive_lexicon_file,encoding)
        self.negatives = self.to_set(negative_lexicon_file,encoding)
        self.tokenize_chars = tokenize_chars
        #TODO use unsupervised tokenization!!!!
        if self.tokenize_chars:  # for chinese
            self.positives = split_patterns(self.positives)
            self.negatives = split_patterns(self.negatives)

    def to_set(self,arg,encoding,weights=None):
        if type(arg) == set:
            return arg
        elif type(arg) == list:
            return set(arg)
        elif type(arg) == str:
            return load_ngrams(arg,encoding,weights=weights)
        return null
    
    def get_sentiment(self,text,context=None,debug=False):
        wordcnt = len(text.strip().split(' '))
        itemcnt = 1
    
        pos, neg, sen = self.get_sentiment_words(text,debug=debug)
        con = round(math.sqrt(pos * -neg),2)
            
        return sen, pos, neg, con, wordcnt, itemcnt

    def is_scrub(self,s):
        """
        //TODO: make min word length language-specific to support Chinese
        if (s.length() <= 1)
            return true;
        //TODO: move dash check to parser or replace dashes with scrubsymbols?
        if (Array.containsOnly(s, AL.dashes))
            return true;
        for (int l = 0; l < langs.length; l++)
            //if (Array.contains(langs[l].scrubs,s))
            if (langs[l].isScrub(s))
                return true;
        return false;
        """
        return True if len(s) < 1 or s in self.punct or s in self.scrub else False

    def get_sentiment_words(self,input_text,pc=None,nc=None,rounding=2,debug=False):
        """
        See original reference implementation:
        https://github.com/aigents/aigents-java/blob/master/src/main/java/net/webstructor/data/LangPack.java#L355
        """
        if input_text is None or len(input_text) < 1:
            return 0, 0, 0
        seq = [*input_text] if self.tokenize_chars else tokenize_re(input_text) #TODO unsupervised tokenization
        if debug:
            print("<{}>".format(str(seq)))
        if len(seq) < 1:
            return 0, 0, 0
        p = 0
        n = 0
        N = self.gram_arity
        while N >= 1:
            seq_ngrams = build_ngrams(seq,N)
            if debug:
                print(N,str(seq_ngrams))
            if len(seq_ngrams) > 0:
                for i in range(0,len(seq_ngrams)):
                    w = seq_ngrams[i]
                    if debug:
                        print(N,w)
                    if w is None or len(w) == 0 or (len(w) == 1 and (w[0] is None or self.is_scrub(w[0]))): #some may be None being consumed earlier
                        i += 1
                        continue
                    found = False 
                    if w in self.positives:
                        p += N #weighted
                        if not pc is None:
                            pc.append(w);
                        found = True;
                    elif w in self.negatives:
                        n += N #weighted
                        if not nc is None:
                            nc.append(w);
                        found = True;
                    if found:
                        for Ni in range(0,N):
                            seq[i + Ni] = None
                        i += N
                    else:
                        i += 1            
            N -= 1
        lenseq = len(seq)
        if self.sentiment_logarithmic:
            p = math.log10(1 + 100 * p / lenseq)/2
            n = math.log10(1 + 100 * n / lenseq)/2
        else:
            p = p / lenseq
            n = n / lenseq
        c = (p-n)/max(p,n) if self.sentiment_maximized else p-n
        if not rounding is None:
            p = round(p,rounding)
            n = round(n,rounding)
            c = round(c,rounding)
        return p, -n, c
    

class TextMetrics(PygentsSentiment):
    def __init__(self, metrics, metric_logarithmic=True, tokenize_chars=False, scrub=[], encoding=None, weighted=False, debug=False):
        self.metric_logarithmic = metric_logarithmic
        self.tokenize_chars = tokenize_chars
        self.scrub = scrub
        self.punct = set(list(punct))
        self.gram_arity = 1
        self.metrics = {}
        self.weights = {} if weighted else None
        for metric in metrics:
            #TODO use unsupervised tokenization!!!!
            if weighted:
                self.weights[metric] = {}
            ngrams = self.to_set(metrics[metric],encoding,self.weights[metric] if weighted else None)
            if self.tokenize_chars: # for chinese
                ngrams = split_patterns(ngrams)
            self.metrics[metric] = ngrams
            for ngram in ngrams: # get real arity on ngrams
                l = len(ngram)
                if self.gram_arity < l:
                    self.gram_arity = l
 
 
    def get_sentiment_words_markup(self, input_text, lists=None, rounding=2, tokenize = tokenize_re, punctuation = None, priority = True, markup = False, metrics=None, debug=False):
        """
        See original reference implementation:
        https://github.com/aigents/aigents-java/blob/master/src/main/java/net/webstructor/data/LangPack.java#L355
        """
        if metrics is None:
            metrics = self.metrics
        if input_text is None or len(input_text) < 1:
            return (0, 0, 0), None
        #seq = [*input_text] if self.tokenize_chars else tokenize_re(input_text) #TODO unsupervised tokenization
        if self.tokenize_chars:
            seq = [*input_text]
        else:
            seq = [t for t in tokenize(input_text) if not (t in punctuation or t.isnumeric())] if not punctuation is None else tokenize(input_text)
        if markup:
            backup = seq.copy()
        if debug:
            print("<{}>".format(str(seq)))
        if len(seq) < 1:
            return 0, 0, 0
        counts = {}
        N = self.gram_arity
        while N >= 1:
            seq_ngrams = build_ngrams(seq,N)
            if debug:
                print(N,str(seq_ngrams))
            if len(seq_ngrams) > 0:
                for i in range(0,len(seq_ngrams)):
                    w = seq_ngrams[i]
                    if debug:
                        print(N,w)
                    if w is None or len(w) == 0 or (len(w) == 1 and (w[0] is None or self.is_scrub(w[0]))): #some may be None being consumed earlier
                        i += 1
                        continue
                    found = False 
                    
                    for metric in metrics:
                        ngrams = self.metrics[metric]
                        if w in ngrams:         
                            weight = N if self.weights is None else N * self.weights[metric][w]                
                            dictcount(counts,metric,weight)
                            if not lists is None:
                                if not metric in lists:
                                    l = set() # unique set
                                    lists[metric] = l
                                else:
                                    if type(lists[metric]) == list:
                                        lists[metric] = set(lists[metric])
                                    l = lists[metric]
                                l.add(w) # unique set
                            found = True
                    # remove tokens that have matched to N-gram with order n=K, so they are not attempted to get matched to N-grams with n < K
                    # (that is why we iterate them from n=N_max down to n=1)
                    # For example, the text "the weather is not good today", after matching ["not", "good"] with n=2,
                    # turns into "the weather is None None today",
                    # so that ["not"] and ["good"] are not counted any more with n=1
                    if found:
                        if priority:
                            for Ni in range(0,N):
                                seq[i + Ni] = None
                        i += N
                    else:
                        i += 1  
            N -= 1
        lenseq = len(seq)
        
        if self.metric_logarithmic:
            for metric in counts:
                counts[metric] = math.log10(1 + 100 * counts[metric] / lenseq)/2
        else:
            for metric in counts:
                counts[metric] = counts[metric] / lenseq

        if not rounding is None:
            for metric in counts:
                counts[metric] = round(counts[metric],rounding)
        if 'positive' in counts and 'negative' in counts:
            counts['contradictive'] = round(math.sqrt(counts['positive'] * counts['negative']),2)

        if not lists is None: # decoded tuples to lists
            for l in lists:
                lists[l] = list(lists[l])

        if markup:
            return counts, markup_blocks(backup,seq)
        else:
            return counts, None

    def get_sentiment_words(self, input_text, lists=None, rounding=2, tokenize = tokenize_re, punctuation = None, priority = True, debug=False):
        return self.get_sentiment_words_markup(input_text, lists, rounding, tokenize, punctuation, priority, False, None, debug)[0] # markup=False, metrics=None

def markup_words(backup,tagged):
    marked_str = ""
    for backup_token, actual_token in zip(backup,tagged):
        token = "__" + backup_token + "__" if actual_token is None else backup_token # underscore tagged
        if not marked_str:
            marked_str = token
        else:
            marked_str += ' ' + token
    return marked_str

def markup_blocks(backup,tagged):
    marked_str = ""
    state_tagged = False
    for backup_token, actual_token in zip(backup,tagged):
        token_tagged = actual_token is None
        #print(token_tagged,state_tagged)
        if state_tagged != token_tagged: # state changed
            if state_tagged:
                marked_str += '__' # close preivios state, if needed
        if marked_str: # and space if needed
            marked_str += ' '
        if state_tagged != token_tagged: # state changed
            if token_tagged:
                marked_str += '__' # open new state, if needed
        marked_str += backup_token
        state_tagged = token_tagged
    if state_tagged:
        marked_str += '__' # close preivios state, if needed
    return marked_str

def create_int_defaultdict():
    return defaultdict(int)

class Learner:

    def __init__(self, n_max = 4, selection_metrics=('F','UF','FN','TF-IDF','UFN','UFN/D/D','FN*UFN','FN*UFN/D','NLMI','FCR','CFR','MR') ):       
        self.labels = defaultdict(int) # A dictionary of label/category counts
        
        # Creating dictionaries for counting n-grams
        self.n_gram_dicts = defaultdict(create_int_defaultdict) # A dictionary for each label/category
        self.all_n_grams = defaultdict(int)  # A general dictionary for all n-grams,
        # self.doc_counts = defaultdict(int) # A count of documents mentioning the n-gram uniquely

        self.uniq_n_gram_dicts = defaultdict(create_int_defaultdict) # Counts of uniq N-grams by label/category
        self.uniq_all_n_grams = defaultdict(int)  # A general dictionary for all n-grams uniq by text (same as self.doc_counts)
        self.n_gram_labels = defaultdict(create_int_defaultdict) # Counts of labels/categories by N-gram
        # self.data_len = 0  # number of documents
        self.n_max = n_max # n_gram max length (do not pass as a "learn" argument or remove at all?)
        self.selection_metrics = selection_metrics
    
    def count_labels(self,labels):
        for label in labels:
            dictcount(self.labels,label)

    def count_ngrams(self,labels,n_grams):
        dictcount(self.all_n_grams, n_grams)
        for label in labels:
            #print('dict=',self.n_gram_dicts[label],'label=',label)
            dictcount(self.n_gram_dicts[label], n_grams)  # Increment the counter for the corresponding label/category

        uniq_n_grams = set(n_grams)
        for uniq_n_gram in uniq_n_grams:
            # self.doc_counts[uniq_n_gram] += 1
            dictcount(self.uniq_all_n_grams, uniq_n_gram)
            for label in labels:
                dictcount(self.uniq_n_gram_dicts[label], uniq_n_gram)
                dictcount(self.n_gram_labels[uniq_n_gram],label)

    def normalize(self):
        self.metrics = {}
        # F: raw frequency
        self.metrics['F'] = self.n_gram_dicts
        # UF: unique frequency
        self.metrics['UF'] = self.uniq_n_gram_dicts
        # FN
        self.metrics['FN'] = dictdict_div_dict(self.n_gram_dicts, self.all_n_grams)
        if self.selection_metrics == ('FN',): # do nothing else!
            return

        # TF-IDF is computed on principle "n-grams vs. lables/categories" rather than "words vs. texts"
        tfidf = defaultdict(dict)
        N = len(self.n_gram_dicts) # number of labels/categories
        for label, ngram_dict in self.n_gram_dicts.items():
            for n_gram, count in ngram_dict.items():
                tf = self.metrics['FN'][label][n_gram] # frequency of N-gram per label/category denominated by total frequency of N-gram
                idf = math.log(N / len(self.n_gram_labels[n_gram]) ) # total number of labels/categories (not documents) denominated by number of labels per N-gram
                tfidf[label][n_gram] = tf * idf
        self.metrics['TF-IDF'] = tfidf

        # UFN: unique frequency normalized
        self.metrics['UFN'] = dictdict_div_dict(self.uniq_n_gram_dicts, self.uniq_all_n_grams)
        # UF/D/D: UF divided by doc counts
        uniq_n_gram_dicts = self.metrics['UF']
        norm_uniq_n_gram_dicts = {}
        for uniq_n_gram_dict in uniq_n_gram_dicts: # iterate over all labels
            norm_uniq_n_gram_dict = {}
            norm_uniq_n_gram_dicts[uniq_n_gram_dict] = norm_uniq_n_gram_dict
            dic = uniq_n_gram_dicts[uniq_n_gram_dict] # pick uniq count of ngrams per labels
            for n_gram in dic:
                #if len(n_gram) <= self.n_max: # TODO remove with assert later!?
                    assert(len(n_gram) <= self.n_max)
                    norm_uniq_n_gram_dict[n_gram] = float( dic[n_gram] ) / self.labels[uniq_n_gram_dict] / len(self.n_gram_labels[n_gram])
        self.metrics['UFN/D/D'] = norm_uniq_n_gram_dicts
        # FN*UFN
        fn = self.metrics['FN']
        ufn = self.metrics['UFN']
        self.metrics['FN*UFN'] = dictdict_mul_dictdict(fn, ufn)
        # FN*UFN/D
        n_gram_labels_counts = {}
        for n_gram, label_dict in self.n_gram_labels.items():
            n_gram_labels_counts[n_gram] = len(label_dict)
        self.metrics['FN*UFN/D'] = dictdict_div_dict(self.metrics['FN*UFN'], n_gram_labels_counts)
        # NLMI, FCR, CFR, MR
        nl_mi = {}
        for label in self.uniq_n_gram_dicts:
            dic = self.uniq_n_gram_dicts[label]
            nl_mi[label] = {}
            for n_gram in dic:
                nl_mi[label][n_gram] = dic[n_gram] * dic[n_gram] / (self.labels[label] * self.uniq_all_n_grams[n_gram])    
        fcr = {}
        cfr = {}
        mr = {}
        for label in self.uniq_n_gram_dicts:
            dic = self.uniq_n_gram_dicts[label]
            fcr[label] = {}
            cfr[label] = {}
            mr[label] = {}
            for n_gram in dic:
                features_by_cat = sum(dic.values()) # features by category
                cats_by_feature = sum(self.n_gram_labels[n_gram].values()) # categories by feature
                fcr[label][n_gram] = dic[n_gram] / cats_by_feature # feature to category relevance - denominated by n of categories by feature
                cfr[label][n_gram] = dic[n_gram] / features_by_cat # category to feature relevance - denominated by n of features by category
                mr[label][n_gram] = dic[n_gram] * dic[n_gram] / (features_by_cat * cats_by_feature)
        self.metrics['NLMI'] = nl_mi
        self.metrics['FCR'] = fcr
        self.metrics['CFR'] = cfr
        self.metrics['MR'] = mr
        return      
    
    def export(self,metric='FN',inclusion_threshold=50,rescale=False):
        return dict_of_dicts_compress_by_threshold(self.metrics[metric],inclusion_threshold,rescale=rescale)

    def save(self,path,name,metric='FN',inclusion_threshold=50,rescale=False):
        model = self.export(metric=metric,inclusion_threshold=inclusion_threshold,rescale=rescale)        
        if not os.path.exists(path):
            os.makedirs(path)
        path += '/'+name
        if not os.path.exists(path):
            os.makedirs(path)  
        for label, ngrams in model.items():
            label_name = label # label.replace(" ", "_")
            file_path = f"{path}/{label_name}.txt"
            sorted_ngrams = sorted(ngrams.items(), key=lambda x: (-x[1],x[0]))
            with open(file_path, "w", encoding="utf-8") as f:
                for ngram, metric_value in sorted_ngrams:
                    ngram_str = ' '.join(ngram)
                    f.write(f"{ngram_str}\t{metric_value}\n")
        
    def learn_sentence(self, text, labels, n_max=4, tokenize = tokenize_re, punctuation = None, debug = False):
        tokens = [t for t in tokenize(text) if not (t in punctuation or t.isnumeric())] if not punctuation is None else tokenize(text)
        for n in range(1, n_max + 1):
            n_grams = build_ngrams(tokens, n)
            self.count_ngrams(labels,n_grams)

    def learn(self, text_labels, n_max=4, tokenize = tokenize_re, punctuation = None, sent=False, debug = False):
        # self.data_len += len(text_labels)
        self.n_max = n_max # use globally defined in constructor
        for text_label in text_labels:
            text = text_label[0]
            labels = text_label[1]
            if debug:
                print(text,labels)
            self.count_labels(labels)
            if sent:
                sentences = nltk.sent_tokenize(text) # this gives us a list of sentences
                if debug:
                    print(sentences)
                for sentence in sentences: # now loop over each sentence and learn it separately
                    self.learn_sentence(sentence, labels, n_max, tokenize, punctuation, debug)    
            else:
                self.learn_sentence(text, labels, n_max, tokenize, punctuation, debug)

        self.normalize()
        return self
