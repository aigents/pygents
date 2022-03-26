import html
import pandas as pd
import urllib.request
from pygents.util import dictcount, merge_two_dicts, countcount, counters_init, merge_dicts

def url_text(url,debug = False):
    text = ''
    lines = 0
    for line in urllib.request.urlopen(url):
        utf8 = line.decode('utf-8')
        #print(utf8)
        lines += 1
        text += utf8.replace('\r',' ').replace('\n','')
    if debug:
        print(lines)
    return text

def url_text_lines(url,debug = False):
    lines = []
    for line in urllib.request.urlopen(url):
        utf8 = line.decode('utf-8')
        lines.append( utf8.replace('\r',' ').replace('\n','') )
    return lines

def text_lines_sample(text_lines,required_count,excluded_prefixes):
    delta = round(len(text_lines)/(required_count+1)) # advance 1 to enable skips within the delta 
    sample = []
    for i in range(required_count):
        line = i * delta
        while True:
            text = text_lines[line].lower()
            if len(text) == 0 or text.startswith(tuple(excluded_prefixes)) and line+1 < len(text_lines): # skip
                line += 1
            else:
                sample.append(text)
                break
    return sample

def unescape_text(text):
    text = html.unescape(text) # &amp;#x200B; => &#x200B;
    text = html.unescape(text) # &amp;#x200B; =>  
    return text

def preprocess_text(text):
    text = html.unescape(text) # &amp;#x200B; => &#x200B;
    text = html.unescape(text) # &amp;#x200B; =>  
    return text.lower()

def grams_init(max_n):  
    return [{} for n in range(max_n)]

def grams_count(counter,chars,n):
    freqs = counter[n-1]
    #print(chars,n)
    for i in range(len(chars) - (n-1)):
        gram = None
        for j in range(n):
            gram = chars[i+j] if gram is None else gram + chars[i+j]
            #print(i,j,gram)
        dictcount(freqs,gram)

def text_grams_count(counter,text,max_n):
    chars = list(text)
    for n in range(max_n):
        grams_count(counter,chars,n+1)

def tokenize_with_lexicon(alphalex,text):
    alex = list(alphalex) #TODO precompile
    alex.sort(key=len,reverse=True)
    tokens = []
    start = 0
    cur = 0
    length = len(text)
    while cur < length:
        subtext = text[cur:]
        #print(subtext)
        for al in alex:
            matched = False
            if subtext.startswith(al):
                if start < cur:
                    tokens.append(text[start:cur])
                tokens.append(al)
                cur += len(al)
                start = cur
                matched = True
                break
        if not matched:
            cur += 1
                
    if start < cur:
        tokens.append(text[start:cur])
    return tokens

def get_grams(text,n):
    grams = []
    chars = list(text)
    for i in range(len(chars) - (n-1)):
        gram = None
        for j in range(n):
            gram = chars[i+j] if gram is None else gram + chars[i+j]
        grams.append(gram)
    return grams

def print_grams(counter,text,n):
    grams = get_grams(text,n)
    freqs = counter[n-1]
    for gram in grams:
        print(gram,freqs[gram])

def merge_grams_at(grams,pos):
    length = len(grams)
    if pos == 0:
        return [grams[0]+grams[1][1:]] + grams[2:]
    elif pos == length - 1:
        return grams[:-2] + [grams[length - 2]+grams[length - 1][1:]]
    else:
        return grams[:pos-1] + [  grams[pos-1][:-1] + grams[pos] + grams[pos+1][1:] ] + grams[pos+2:] 
        
        
def merge_grams(counter,text,n_start,n_cycles):
    grams = get_grams(text,n_start)
    for cycle in range(n_cycles):
        max_freq = 0
        max_pos = 0
        for pos in range(len(grams)):
            gram = grams[pos]
            freq = counter[gram] if gram in counter else 0
            print(gram,freq)
            if max_freq < freq:
                max_pos = pos
                max_freq = freq
        print(cycle,'==>',grams[max_pos],max_freq)
        grams = merge_grams_at(grams,max_pos)

# Bigram counts to Mutual Information 
# https://arxiv.org/pdf/cmp-lg/9805009.pdf
# page 40
def counts2mis(bigram_counts,debug=False):
    mis = {}
    n_x_ = {}
    n__y = {}
    n = 0
    for bigram in bigram_counts:
        x = bigram[0]
        y = bigram[1]
        n_xy = bigram_counts[bigram]
        n += n_xy
        dictcount(n_x_,x,n_xy)
        dictcount(n__y,y,n_xy)
    #c = 0
    for bigram in bigram_counts:
        x = bigram[0]
        y = bigram[1]
        n_xy = bigram_counts[bigram]
        if debug:
            print(bigram, x, y, n_xy, n, n_x_[x] , n__y[y])
        mis[bigram] = n_xy * n / (n_x_[x] * n__y[y])
        #c+=1
        #if c > 10:
        #    break
    return mis


def grams_count_with_char_freedoms(gram_counter,forth_freedom_counter,back_freedom_counter,chars,n,debug=False):
    freqs = gram_counter[n-1]
    back_freedoms = back_freedom_counter[n-1]
    forth_freedoms = forth_freedom_counter[n-1]
    #print(chars,n)
    length = len(chars)
    for i in range(length - (n-1)):
        gram = None
        for j in range(n):
            gram = chars[i+j] if gram is None else gram + chars[i+j]
        if debug:
            print(gram)
        dictcount(freqs,gram)
        if i < (length - n):
            if debug:
                print('+',gram,chars[i+n])
            countcount(forth_freedoms,gram,chars[i+n])
        if i > 0:
            if debug:
                print('-',gram,chars[i-1])
            countcount(back_freedoms,gram,chars[i-1])
        
def model_grams_count_with_char_freedoms(texts,max_n,debug=False):
    model = counters_init(max_n) 
    for text in texts:
        text = preprocess_text(text)
        chars = list(text)
        for n in range(max_n):
            grams_count_with_char_freedoms(model[0],model[1],model[2],chars,n+1)
    return [merge_dicts(d) for d in model]

 
#kinda unit test
_test_counters1 = counters_init(2)
assert str(_test_counters1) == '([{}, {}], [{}, {}], [{}, {}])'
grams_count_with_char_freedoms(_test_counters1[0],_test_counters1[1],_test_counters1[2],list("abaxb"),1)
grams_count_with_char_freedoms(_test_counters1[0],_test_counters1[1],_test_counters1[2],list("abaxb"),2)
assert str([merge_dicts(d) for d in _test_counters1]) == "[{'a': 2, 'b': 2, 'x': 1, 'ab': 1, 'ba': 1, 'ax': 1, 'xb': 1}, {'a': {'b': 1, 'x': 1}, 'b': {'a': 1}, 'x': {'b': 1}, 'ab': {'a': 1}, 'ba': {'x': 1}, 'ax': {'b': 1}}, {'b': {'a': 1, 'x': 1}, 'a': {'b': 1}, 'x': {'a': 1}, 'ba': {'a': 1}, 'ax': {'b': 1}, 'xb': {'a': 1}}]"
assert str(model_grams_count_with_char_freedoms(["abaxb"],2)) == "[{'a': 2, 'b': 2, 'x': 1, 'ab': 1, 'ba': 1, 'ax': 1, 'xb': 1}, {'a': {'b': 1, 'x': 1}, 'b': {'a': 1}, 'x': {'b': 1}, 'ab': {'a': 1}, 'ba': {'x': 1}, 'ax': {'b': 1}}, {'b': {'a': 1, 'x': 1}, 'a': {'b': 1}, 'x': {'a': 1}, 'ba': {'a': 1}, 'ax': {'b': 1}, 'xb': {'a': 1}}]"
assert str(model_grams_count_with_char_freedoms(["abaxb","abaxb"],2)) == "[{'a': 4, 'b': 4, 'x': 2, 'ab': 2, 'ba': 2, 'ax': 2, 'xb': 2}, {'a': {'b': 2, 'x': 2}, 'b': {'a': 2}, 'x': {'b': 2}, 'ab': {'a': 2}, 'ba': {'x': 2}, 'ax': {'b': 2}}, {'b': {'a': 2, 'x': 2}, 'a': {'b': 2}, 'x': {'a': 2}, 'ba': {'a': 2}, 'ax': {'b': 2}, 'xb': {'a': 2}}]"


def grams_count_with_gram_freedoms(counters,text,n,debug=False):
    freqs = counters[0][n-1]
    forth_freedoms = counters[1][n-1]
    back_freedoms = counters[2][n-1]
    #print(chars,n)
    length = len(text)
    for i in range(length - (n-1)):
        #count grams
        gram = text[i:i+n]
        if debug:
            print("\t{}".format(gram))
        dictcount(freqs,gram)
        #count backs
        #"""
        if i > 0:
            back = i - n
            if back < 0:
                back = 0
            gram_back = text[back:i]
            if debug:
                print("-\t{}\t{}".format(gram,gram_back))
            countcount(back_freedoms,gram,gram_back)
        #"""
        #count forths
        if i < (length - n):
            forth = i + n + n
            if forth > length:
                forth = length
            gram_forth = text[i+n:forth]
            if debug:
                print("+\t{}\t{}".format(gram,gram_forth))
            countcount(forth_freedoms,gram,gram_forth)
#kinda unit test
_test_m = counters_init(1)
grams_count_with_gram_freedoms(_test_m,"abcd",1,debug=False)            
#print(_test_m[0])
#print(_test_m[1])
#print(_test_m[2])
#print(_test_m)
assert str(_test_m) == "([{'a': 1, 'b': 1, 'c': 1, 'd': 1}], [{'a': {'b': 1}, 'b': {'c': 1}, 'c': {'d': 1}}], [{'b': {'a': 1}, 'c': {'b': 1}, 'd': {'c': 1}}])"
#print()
_test_m = counters_init(2)
grams_count_with_gram_freedoms(_test_m,"abcde",2,debug=False)            
#print(_test_m[0])
#print(_test_m[1])
#print(_test_m[2])
#print(_test_m)
assert str(_test_m) == "([{}, {'ab': 1, 'bc': 1, 'cd': 1, 'de': 1}], [{}, {'ab': {'cd': 1}, 'bc': {'de': 1}, 'cd': {'e': 1}}], [{}, {'bc': {'a': 1}, 'cd': {'ab': 1}, 'de': {'bc': 1}}])"
_test_m = counters_init(3)
grams_count_with_gram_freedoms(_test_m,"abcdef",3,debug=False)            
#print(_test_m[0])
#print(_test_m[1])
#print(_test_m[2])
#print(_test_m)
assert str(_test_m) == "([{}, {}, {'abc': 1, 'bcd': 1, 'cde': 1, 'def': 1}], [{}, {}, {'abc': {'def': 1}, 'bcd': {'ef': 1}, 'cde': {'f': 1}}], [{}, {}, {'bcd': {'a': 1}, 'cde': {'ab': 1}, 'def': {'abc': 1}}])"


def profile_probabilities(counters,text,max_n,debug=False):
    length = len(text)
    de_list = []
    for i in range(1,length-1):
        sym = text[i]
        #forward
        start = i - (max_n-1)
        if start < 0:
            start = 0
        pre_gram = text[start:i]
        post_gram = text[start:i+1]
        pre_prob = counters[pre_gram] if pre_gram in counters else 1
        post_prob = counters[post_gram] if post_gram in counters else 0
        prob = post_prob / pre_prob
        #backward
        end = i + max_n
        if end > length:
            end = length
        back_pre_gram = text[i+1:end]
        back_post_gram = text[i:end]
        back_pre_prob = counters[back_pre_gram] if back_pre_gram in counters else 1
        back_post_prob = counters[back_post_gram] if back_post_gram in counters else 0
        back_prob = back_post_prob / back_pre_prob

        if debug:
            print("Forw {}-{}:'{}'-'{}'={}=>{}".format(start,i,post_gram,pre_gram,sym,prob))
            print("Back {}-{}:'{}'-'{}'={}=>{}".format(i,end,back_post_gram,back_pre_gram,sym,back_prob))
        de_list.append((i,sym,prob,back_prob))
    return de_list


# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2655800/
def profile_freedoms(model,text,max_n,debug=False):
    length = len(text)
    de_list = []
    for i in range(length):
        sym = text[i]
        #forward
        counters = model[1]
        start = i - (max_n-1)
        if start < 0:
            start = 0
        #gram = text[start:i]
        forw_gram = text[start:i+1]
        forw_freedom = len(counters[forw_gram]) if forw_gram in counters else 0
        #backward
        counters = model[2]
        end = i + max_n
        if end > length:
            end = length
        #gram = text[i+1:end]
        back_gram = text[i:end]
        back_freedom = len(counters[back_gram]) if back_gram in counters else 0
        if debug:
            print("+{}-{}:\t'{}'=>{}\t{}\t-{}-{}:\t'{}'=>{}\t{}".format(start,i,forw_gram,sym,forw_freedom,i,end,back_gram,sym,back_freedom))
        de_list.append((i,sym,forw_freedom,back_freedom))
    return de_list


def profile_freedoms_df(model,text,n,debug=False):
    df = pd.DataFrame(profile_freedoms(model,text,n,debug=debug),columns=['pos','char','f+','f-'])
    df['f+|f-']=df['f+'] + df['f-']
    df['f+&f-']=df['f+'] * df['f-']
    df['df+'] = df['f+'].diff().shift(-1)
    df['df-'] = df['f-'].diff()
    df['df-df+'] = df['df-'] - df['df+']
    return df

