import abc
import pickle
import re
import math
import pandas as pd
import jieba

from pygents.util import count_subelements, dictcount, calc_f1, counters_init, remove_all, dict_update, dict_compress_with_loss 
from pygents.text import preprocess_text, grams_count_with_char_freedoms, grams_count_with_gram_freedoms
from pygents.text import url_lines, tokenize_with_sorted_lexicon, profile_freedoms, profile_probabilities


# Basic Tokenizer
class Tokenizer(abc.ABC):

    def __init__(self, debug=True):
        self.debug = debug

    def tokenize(self,text):
        return text.split()

assert str(Tokenizer().tokenize("ab c")) == "['ab', 'c']"

def tokenize_detaching_head(text,chars="'\"{[("):
    tokens = []
    for head in range(len(text)):
        found = chars.find(text[head])
        if found >= 0:
            tokens.append(chars[found])
        else:
            return tokens, text[head:]
    return tokens, None
assert str(tokenize_detaching_head("test")) == "([], 'test')"
assert str(tokenize_detaching_head("'\"")) == '(["\'", \'"\'], None)'
assert str(tokenize_detaching_head("\"'test")) == "(['\"', \"'\"], 'test')"


def tokenize_detaching_tail(text,chars="'\":,;.!?}])"):
    tokens = []
    length = len(text)
    for i in range(length):
        tail = length - i - 1
        found = chars.find(text[tail])
        if found >= 0:
            tokens.append(chars[found])
        else:
            tokens.reverse()
            return tokens, text[:tail + 1]
    tokens.reverse()
    return tokens, None
assert str(tokenize_detaching_tail("test")) == "([], 'test')"
assert str(tokenize_detaching_tail("test'")) == "([\"'\"], 'test')"
assert str(tokenize_detaching_tail("test.\"")) == "(['.', '\"'], 'test')"
assert str(tokenize_detaching_tail("test').\"")) == "([\"'\", ')', '.', '\"'], 'test')"

    
def tokenize_split_with_delimiters_and_quotes(text):
    tokens = []
    splits = text.split(' ') # TODO add ALL whitespaces like \n, \r \t, etc. 
    for split in splits:
        if len(tokens) > 0:
            tokens.append(' ')
        head, token = tokenize_detaching_head(split)
        tokens.extend(head)
        if token is not None and len(token) > 0: 
            tail, token = tokenize_detaching_tail(token)
            if token is not None and len(token) > 0:
                tokens.append(token)
            tokens.extend(tail)
    return tokens
assert str(tokenize_split_with_delimiters_and_quotes("man says hi")) == "['man', ' ', 'says', ' ', 'hi']"
assert str(tokenize_split_with_delimiters_and_quotes("man (tom) says 'hi there!' to me.")) == "['man', ' ', '(', 'tom', ')', ' ', 'says', ' ', \"'\", 'hi', ' ', 'there', '!', \"'\", ' ', 'to', ' ', 'me', '.']"


#TODO case sensitivity
class DelimiterTokenizer(Tokenizer):
    def __init__(self):
        Tokenizer.__init__(self,debug=False)
    def tokenize(self,text):
        return tokenize_split_with_delimiters_and_quotes(text)
assert str(DelimiterTokenizer().tokenize("man (tom) says 'hi there!' to me.")) == "['man', ' ', '(', 'tom', ')', ' ', 'says', ' ', \"'\", 'hi', ' ', 'there', '!', \"'\", ' ', 'to', ' ', 'me', '.']"
assert str(DelimiterTokenizer().tokenize("hi there, man!")) == "['hi', ' ', 'there', ',', ' ', 'man', '!']"
assert str(DelimiterTokenizer().tokenize("man says: hi there!, man.")) == "['man', ' ', 'says', ':', ' ', 'hi', ' ', 'there', '!', ',', ' ', 'man', '.']"


#https://github.com/fxsjy/jieba
class JiebaTokenizer(Tokenizer):
    def __init__(self):
        Tokenizer.__init__(self,debug=False)
    def tokenize(self,text):
        return [r[0] for r in jieba.tokenize(text)]


# Lexicon-based Tokenization

class LexiconTokenizer(Tokenizer): # very unefficient because of iterative search, use indexed LexiconIndexedTokenizer instead

    def __init__(self, name=None, lexicon=None, cased=False, url=None, debug=False):
        Tokenizer.__init__(self,debug=debug)
        self.name = name
        if not lexicon is None: 
            self.alex = list(lexicon) #copy
        else:
            lex_lines = url_lines(url)
            self.alex = [re.split('\t| |,|;|\n|\r',line)[0] for line in lex_lines] #load from url
            # TODO load from file
        self.compile()
        self.cased = cased

    def compile(self):
        self.alex.sort(key=len,reverse=True) #precompile

    def tokenize(self,text):
        return tokenize_with_sorted_lexicon(self.alex,text,cased=self.cased)

assert str(LexiconTokenizer(lexicon=['tuna','is','fish','cat','mammal']).tokenize("tunaisafish.catisamammal"))=="['tuna', 'is', 'a', 'fish', '.', 'cat', 'is', 'a', 'mammal']"    
assert str(LexiconTokenizer(lexicon=['tuna','is','fish','cat','mammal']).tokenize("Tunaisafish.Catisamammal"))=="['Tuna', 'is', 'a', 'fish', '.Cat', 'is', 'a', 'mammal']"
assert str(LexiconTokenizer(lexicon=['tuna','is','fish','cat','mammal'],cased=True).tokenize("Tunaisafish.Catisamammal"))=="['Tuna', 'is', 'a', 'fish', '.', 'Cat', 'is', 'a', 'mammal']"


def prefixed_match_from_list(lst,text):
    for item in lst:
        if text.startswith(item[0]):
            return item
    return None

def prefixed_match(prefixed_dict,text):
    letter = text[0]
    if not letter in prefixed_dict:
        return None
    return prefixed_match_from_list(prefixed_dict[letter],text)

def tokenize_with_prexied_sorted_lexicon(prefixed_dict,text,cased=False):
    original = text
    if cased: #if need to spend time on lowercasing non-lowercased text
        text = text.lower()
    tokens = []
    start = 0
    cur = 0
    length = len(text)
    sum_weight = 0
    while cur < length:
        subtext = text[cur:]
        word_weight = prefixed_match(prefixed_dict,subtext)
        #print(al)
        if not word_weight is None:
            word_len = len(word_weight[0])
            if start < cur:
                tokens.append(original[start:cur])
            tokens.append(original[cur:cur+word_len])
            sum_weight += word_weight[1]
            cur += word_len
            start = cur
        else:
            cur += 1
            #print('yo')
    if start < cur:
        tokens.append(original[start:cur])
        #print(original[start:cur])
    return tokens, sum_weight

def tabbed_line2tuple(line,log=True):
    lst = re.split('\t| |,|;|\n|\r',line)
    if len(lst) > 1:
        return (lst[0],float(lst[1]) if not log else math.log10(1+float(lst[1])))
    else:
        return (lst[0],1.0)

def weightedlist2dict(lst,lower=False): # (key,weight) -> sum weigts by keys, keys may be lowercased
    dic = {}
    for item in lst:
        dictcount(dic,item[0].lower() if lower else item[0],item[1])
    return dic

class LexiconIndexedTokenizer(Tokenizer):

    def __init__(self, name=None, lexicon=None, cased=False, debug=False, url=None, sortmode=0):
        Tokenizer.__init__(self,debug=debug)
        self.name = name
        if not lexicon is None: 
            self.freqlist = [(word,1.0) for word in lexicon] if type(lexicon[0]) is str else lexicon #copy
        elif not url is None:
            lex_lines = url_lines(url)
            self.freqlist = [tabbed_line2tuple(line) for line in lex_lines] #load from url
            # TODO load from file
        self.sortmode = sortmode
        self.compile()
        self.cased = cased

    def compile(self):
        self.dict = {}
        self.fulldict = weightedlist2dict(self.freqlist,lower=True) # save for debugging only!?
        for key in self.fulldict:
            value = self.fulldict[key]
            if len(key) > 0:
                letter = key[0]
                if not letter in self.dict:
                    self.dict[letter] = set()
                self.dict[letter].add((key,value))
        #print(self.dict['f'])
        for key in self.dict:
            lst = list(self.dict[key])
            if self.sortmode == 0: # by len
                lst.sort(key=lambda s: len(s[0]), reverse=True)
            elif self.sortmode == 1: # by weight
                lst.sort(key=lambda s: s[1], reverse=True)
            else: # by len times logweight
                #TODO log separately for better performance
                lst.sort(key=lambda s: math.log10(s[1])*len(s[0]), reverse=True)
            self.dict[key] = lst
        #print(self.dict['f'])

    def tokenize(self,text):
        tokens, weight = tokenize_with_prexied_sorted_lexicon(self.dict,text,cased=self.cased)
        return tokens

    def tokenize_weight(self,text):
        tokens, weight = tokenize_with_prexied_sorted_lexicon(self.dict,text,cased=self.cased)
        length = len(tokens)
        return tokens, 0 if length == 0 else weight / length 

    def count_params(self):
        return len(self.freqlist)

assert str(LexiconIndexedTokenizer(lexicon=['tuna','is','fish','cat','mammal']).tokenize("tunaisafish.catisamammal"))=="['tuna', 'is', 'a', 'fish', '.', 'cat', 'is', 'a', 'mammal']"    
assert str(LexiconIndexedTokenizer(lexicon=['tuna','is','fish','cat','mammal']).tokenize("Tunaisafish.Catisamammal"))=="['Tuna', 'is', 'a', 'fish', '.Cat', 'is', 'a', 'mammal']"
assert str(LexiconIndexedTokenizer(lexicon=['tuna','is','fish','cat','mammal'],cased=True).tokenize("Tunaisafish.Catisamammal"))=="['Tuna', 'is', 'a', 'fish', '.', 'Cat', 'is', 'a', 'mammal']"


# Extended Tokenizer based on "freedoms"
class FreedomTokenizer(Tokenizer):

    def __init__(self, name=None, max_n=7, mode='grams', debug=False):
        Tokenizer.__init__(self,debug=debug)
        self.max_n = max_n
        self.model = pickle.load(open(name, 'rb')) if name is not None else [{},{},{}]
        self.mode = mode

    def train(self,texts,max_n=None):
        if max_n is None:
            max_n = self.max_n
        model = counters_init(max_n) 
        for text in texts:
            text = preprocess_text(text)
            if self.mode == 'grams':
                for n in range(max_n):
                    grams_count_with_gram_freedoms(model,text,n+1,debug=self.debug)
            else: # 'chars' - legacy, woorks better on Brown corpus!
                chars = list(text)
                for n in range(max_n):
                    grams_count_with_char_freedoms(model[0],model[1],model[2],chars,n+1,debug=self.debug)
        #merge n-specific models into joint ones
        for i in range(3):
            for d in model[i]:
                #self.model[i].update(d)
                dict_update(self.model[i],d)
        return self

    def train_folder(self,folder_path,model_path=None,name=None,debug = False):
        #TODO recursion, if specified
        onlyfiles = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]
        cnt = 0
        for file in onlyfiles:
            with open(join(path, file),errors='ignore') as f:
                lines = f.readlines()
                cnt += 1
                if debug and (cnt % 100) == 0:
                    print(cnt,file)
                self.train(lines)
        if model_path is not None:
            self.store(model_path)
        if debug:
            print(self.count_params())
        
    def tokenize(self,text):
        #TODO pass!!!???
        return text.split()

    def count_params(self):
        return count_subelements(self.model)
    
    def store(self,path):
        pickle.dump(self.model, open(path, 'wb'), pickle.HIGHEST_PROTOCOL)

    
_test_tokenizer = FreedomTokenizer(max_n=2,mode='chars',debug=False).train(["pig"])
assert _test_tokenizer.count_params() == 11
assert str(_test_tokenizer.model) == "[{'p': 1, 'i': 1, 'g': 1, 'pi': 1, 'ig': 1}, {'p': {'i': 1}, 'i': {'g': 1}, 'pi': {'g': 1}}, {'i': {'p': 1}, 'g': {'i': 1}, 'ig': {'p': 1}}]"
_test_tokenizer = FreedomTokenizer(max_n=2,mode='chars').train(["ding","dong"])
#print(_test_tokenizer.count_params())
assert _test_tokenizer.count_params() == 28
#print(str(_test_tokenizer.model[0]))
#print(str(_test_tokenizer.model[1]))
#print(str(_test_tokenizer.model[2]))
#print(str(_test_tokenizer.model))
assert str(_test_tokenizer.model) == "[{'d': 2, 'i': 1, 'n': 2, 'g': 2, 'o': 1, 'di': 1, 'in': 1, 'ng': 2, 'do': 1, 'on': 1}, {'d': {'i': 1, 'o': 1}, 'i': {'n': 1}, 'n': {'g': 2}, 'o': {'n': 1}, 'di': {'n': 1}, 'in': {'g': 1}, 'do': {'n': 1}, 'on': {'g': 1}}, {'i': {'d': 1}, 'n': {'i': 1, 'o': 1}, 'g': {'n': 2}, 'o': {'d': 1}, 'in': {'d': 1}, 'ng': {'i': 1, 'o': 1}, 'on': {'d': 1}}]"


class FreedomBasedTokenizer(FreedomTokenizer):

    def __init__(self, base, back, forw, nlist=[1], threshold=0.5, debug=False):
        FreedomTokenizer.__init__(self,debug=debug)
        self.model = base.model
        self.mode = base.mode
        self.back = back
        self.forw = forw
        self.nlist = nlist
        self.threshold = threshold
        
    def set_options(self,**kwargs):
        if 'threshold' in kwargs:
            self.threshold = kwargs['threshold']
        if 'nlist' in kwargs:
            self.nlist = kwargs['nlist']

    def tokenize(self,text):
        return tokenize_with_opposite_metrics(self.model,text,self.back,self.forw,self.nlist,self.threshold)


def model_compress_with_loss(model,threshold=0.01):
    dict_compress_with_loss(model[0],threshold)
    dict_compress_with_loss(model[1],threshold)
    dict_compress_with_loss(model[2],threshold)


def profile_freedoms_ex_df(model,text,n,denominate=False,debug=False):
    df = pd.DataFrame(profile_freedoms(model,text,n,denominate=denominate,debug=debug),columns=['pos','gram','f+','f-'])
    df['ddf+'] = (df['f+'] - df['f+'].mean()).clip(lower=0)
    df['ddf-'] = (df['f-'] - df['f-'].mean()).clip(lower=0)
    df['ddf+|ddf-'] = df['ddf+'] + df['ddf-'].shift(-1)
    df['ddf+&ddf-'] = df['ddf+'] * df['ddf-'].shift(-1)
    df['df+'] = df['f+'].diff() 
    df['df-'] = -df['f-'].diff().shift(-1)
    df['df+|df-'] = df['df+'] + df['df-']
    df['df+&df-'] = df['df+'] * df['df-']
    # We assigned a “peak” value to each character transition, 
    # computed by adding the value of the preceding increase in freedom to the following decrease in freedom. 
    # We characterized token boundaries based on the sum of their forward- and backward-reading peak values.
    df['peak+'] = df['df+'] - df['df+'].shift(-1)
    df['peak-'] = df['df-'] - df['df-'].shift(1)
    df['f+|f-'] = df['f+'] + df['f-'].shift(-1)
    df['f+&f-'] = df['f+'] * df['f-'].shift(-1)
    return df


def profile_freedoms_avg_df(model,text,metrics,nlist,denominate=False,debug=False):
    res_df = None
    for n in nlist:
        df = profile_freedoms_ex_df(model,text,n,denominate=denominate)
        if res_df is None:
            res_df = df[['pos','gram']+metrics].copy()
        else:
            for m in metrics:
                res_df[m] = res_df[m] + df[m]
    for m in metrics:
        res_df[m] = res_df[m]/res_df[m].max()
    return res_df


def profile_probabilities_ex_df(model,text,n,debug=False):
    df = pd.DataFrame(profile_probabilities(model[0],text,n,debug=debug),columns=['pos','gram','p+','p-'])
    if n == 1:
        pmax = max(df['p+'].max(),df['p-'].max())
        df['p+'] = df['p+']/pmax
        df['p-'] = df['p-']/pmax
    df['ddp+'] = (df['p+'] - df['p+'].mean()).clip(lower=0)
    df['ddp-'] = (df['p-'] - df['p-'].mean()).clip(lower=0)
    df['ddp+|ddp-'] = df['ddp+'] + df['ddp-'].shift(-1)
    df['ddp+&ddp-'] = df['ddp+'] * df['ddp-'].shift(-1)
    df['dp+'] = df['p+'].diff() 
    df['dp-'] = -df['p-'].diff().shift(-1)
    df['dp+|dp-'] = df['dp+'] + df['dp-']
    df['dp+&dp-'] = df['dp+'] * df['dp-']
    #TODO!?
    # We assigned a “peak” value to each character transition, 
    # computed by adding the value of the preceding increase in freedom to the following decrease in freedom. 
    # We characterized token boundaries based on the sum of their forward- and backward-reading peak values.
    #df['peak+'] = df['df+'] - df['df+'].shift(-1)
    #df['peak-'] = df['df-'] - df['df-'].shift(1)
    df['p+|p-'] = df['p+'] + df['p-'].shift(-1)
    df['p+&p-'] = df['p+'] * df['p-'].shift(-1)
    return df


def profile_probabilities_avg_df(model,text,metrics,nlist,debug=False):
    res_df = None
    for n in nlist:
        df = profile_probabilities_ex_df(model,text,n)
        if res_df is None:
            res_df = df[['pos','gram']+metrics].copy()
        else:
            for m in metrics:
                res_df[m] = res_df[m] + df[m]
    for m in metrics:
        res_df[m] = res_df[m]/res_df[m].max()
    return res_df


def tokenize_with_opposite_metrics(model,text,back,forw,nlist,threshold=0.5,profiler=profile_freedoms_avg_df,debug=False):
    tokens = []
    token = ''
    df = profiler(model,text,[forw,back],nlist)
    length = len(df)
    for i in range(length):
        iplus1 = i+1
        brk_back = True if iplus1 < length and df.loc[iplus1][back] >= threshold else False
        brk_forw = True if df.loc[i][forw] >= threshold else False
        #token += df.loc[i]['gram']
        token += text[i] # to ensure raw data capitalization
        if debug:
            print("{}\t{}\t{}\t{}\t{}\t{}".format(df.loc[i]['gram'],'-' if brk_back else '', '+' if brk_forw else '',round(df.loc[i][back],2),round(df.loc[i][forw],2),token))
        if len(token) > 0 and (brk_back or brk_forw):
            tokens.append(token)
            token = ''
    if len(token) > 0:
            tokens.append(token)
    return tokens


def tokenize_with_forward_metric(model,text,forw,nlist,threshold=0.5,profiler=profile_freedoms_avg_df,debug=False):
    tokens = []
    token = ''
    df = profiler(model,text,[forw],nlist)
    length = len(df)
    for i in range(length):
        brk_forw = True if df.loc[i][forw] >= threshold else False
        #token += df.loc[i]['gram']
        token += text[i] # to ensure raw data capitalization
        if debug:
            print("{}\t{}\t{}\t{}\t{}".format(df.loc[i]['gram'],'+' if brk_forw else '',round(df.loc[i][back],2),round(df.loc[i][forw],2),token))
        if len(token) > 0 and brk_forw:
            tokens.append(token)
            token = ''
    if len(token) > 0:
            tokens.append(token)
    return tokens


def evaluate_tokenizer_f1(texts,real_tokenizer,test_tokenizer,debug=False):
    avg_f1 = 0
    count = 0
    for text in texts:
        expected = real_tokenizer.tokenize(text)
        tokens = test_tokenizer.tokenize(text)
        f1 = calc_f1(expected,tokens)
        if debug:
            print(text)
            print(round(f1,2),tokens)
        avg_f1 += f1
        count += 1
    return round(avg_f1/count,2)


def evaluate_tokenizer(model,texts,forw,back,nlist,threshold,profiler=profile_freedoms_avg_df,spaces=False,output=False,debug=False):
    if output:
        print("N\tthres.\tF1")
    f1_avg = 0
    for text in texts:
        tokens = tokenize_with_opposite_metrics(model,text,forw,back,nlist,threshold=threshold,profiler=profiler) if back is not None else tokenize_with_forward_metric(model,text,forw,nlist,threshold=threshold,profiler=profiler)
        tokens_ref = tokenize_split_with_delimiters_and_quotes(text)
        if not spaces:
            remove_all(tokens,' ')
            remove_all(tokens_ref,' ')
        f1 = calc_f1(tokens_ref,tokens) 
        f1_avg += f1
        if debug:
            print(f1)
            print(text)
            print(calc_diff(tokens,tokens_ref))
            print(str(tokens_ref))
            print(str(tokens))
            print()
    f1 = round(f1_avg/len(texts),2)
    if output:
        print("{}\t{}\t{}".format(nlist,threshold,f1))
    return nlist,threshold,f1



