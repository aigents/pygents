# MIT License
# 
# Copyright (c) 2015-2023 Aigents®, Anton Kolonin 
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


import requests
import urllib.parse
import math
import json
import re
import emoji
from pygents.text import url_lines
from pygents.util import dictcount


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

quotes_list =  ["'", '"', '“', '”', '*', '(', ')', '[', ']', '<', '>', '#', '^', '@', '~']
delimiters_list = [',', ';', ':', '.', '!', '?']
delimiters_regexp = r' |\n|\t|\r|\[|\]|\(|\)'

def add_token(token,res_list):
    if len(token) == 0:
        return
    if len(token) == 1:
        res_list.append(token)
        return
    first = token[0]
    if first in emoji.UNICODE_EMOJI['en'] or first in quotes_list or first in delimiters_list:
        res_list.append(first)
        add_token(token[1:],res_list)
        return
    last = token[-1]
    if last in emoji.UNICODE_EMOJI['en'] or last in quotes_list or last in delimiters_list:
        add_token(token[0:-1],res_list)
        res_list.append(last)
        return
    if token_valid(token): # and len(token) > 1 !!!
        if len(token) > 2 and token[0].isalpha() and token[-1].isalpha():
            for delimiter in delimiters_list:
                if token.find(delimiter) != -1:
                    tokens = token.split(delimiter)
                    for i in range(0,len(tokens)):
                        if i > 0:
                            res_list.append(delimiter)
                        add_token(tokens[i],res_list)
                    return
        res_list.append(token)

def tokenize_re(text):
    tokens = re.split(delimiters_regexp,text.lower())
    res_list = []
    for token in tokens:
        if len(token) < 1:
            continue
        add_token(token,res_list)
    return res_list

def build_ngrams(seq,N):
    size = len(seq) - N + 1;
    if size < 1:
        return [];
    items = [];
    for i in range(0,size):
        items.append( tuple(seq[i:i+N]) )
    return items

def load_ngrams(file,encoding=None,debug=False):
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
    #ngrams = [tuple(l.split()) for l in lines if len(l) > 0]
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
punct = "-—{([<})]>.,;:?$_.+!?*'\"\\/"

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

    def to_set(self,arg,encoding):
        if type(arg) == set:
            return arg
        elif type(arg) == list:
            return set(arg)
        elif type(arg) == str:
            return load_ngrams(arg,encoding)
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
    def __init__(self, metrics, metric_logarithmic=True, tokenize_chars=False, scrub=[], encoding=None, debug=False):
        self.metric_logarithmic = metric_logarithmic
        self.tokenize_chars = tokenize_chars
        self.scrub = scrub
        self.punct = set(list(punct))
        self.gram_arity = 1
        self.metrics = {}
        for metric in metrics:
            #TODO use unsupervised tokenization!!!!
            ngrams = self.to_set(metrics[metric],encoding)
            if self.tokenize_chars: # for chinese
                ngrams = split_patterns(ngrams)
            self.metrics[metric] = ngrams
            for ngram in ngrams: # get real arity on ngrams
                l = len(ngram)
                if self.gram_arity < l:
                    self.gram_arity = l
 
 
    def get_sentiment_words(self,input_text,lists=None,rounding=2,debug=False):
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
                    
                    for metric in self.metrics:
                        ngrams = self.metrics[metric]
                        if w in ngrams:                         
                            dictcount(counts,metric,N) # p += N #weighted
                            if not lists is None:
                                if not metric in lists:
                                    l = []
                                    lists[metric] = l
                                else:
                                    l = lists[metric]
                                l.append(w);
                                #print(lists)
                            found = True;
#TODO what is that?
                    if found:
                        for Ni in range(0,N):
                            seq[i + Ni] = None
                        i += N
                    else:
                        i += 1  
            N -= 1
        lenseq = len(seq)
        
#TODO fix this
        if self.metric_logarithmic:
            for metric in counts:
                counts[metric] = math.log10(1 + 100 * counts[metric] / lenseq)/2
        else:
            for metric in counts:
                counts[metric] = counts[metric] / lenseq
#TODO overall
        if not rounding is None:
            for metric in counts:
                counts[metric] = round(counts[metric],rounding)
        if 'positive' in counts and 'negative' in counts:
        	counts['contradictive'] = round(math.sqrt(counts['positive'] * counts['negative']),2)
        return counts
    
