import html
import pandas as pd

def unescape_text(text):
    text = html.unescape(text) # &amp;#x200B; => &#x200B;
    text = html.unescape(text) # &amp;#x200B; =>  
    return text

def preprocess_text(text):
    text = html.unescape(text) # &amp;#x200B; => &#x200B;
    text = html.unescape(text) # &amp;#x200B; =>  
    return text.lower()

def count(dic,arg,cnt=1):
    if arg in dic:
        dic[arg] = dic[arg] + cnt
    else:
        dic[arg] = cnt

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
        count(freqs,gram)

def text_grams_count(counter,text,max_n):
    chars = list(text)
    for n in range(max_n):
        grams_count(counter,chars,n+1)

def merge_two_dicts(x, y):
    """Given two dictionaries, merge them into a new dict as a shallow copy."""
    z = x.copy()
    z.update(y)
    return z

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
        count(n_x_,x,n_xy)
        count(n__y,y,n_xy)
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


def countcount(dic,arg,subarg,cnt=1):
    if arg in dic:
        subdic = dic[arg]
    else:
        dic[arg] = subdic = {}
    count(subdic,subarg,cnt)

def counters_init(max_n):  
    return [{} for n in range(max_n)], [{} for n in range(max_n)], [{} for n in range(max_n)]

def merge_dicts(dicts):
    merged = {}
    for d in dicts:
        merged.update(d)
    return merged

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
        count(freqs,gram)
        if i < (length - n):
            if debug:
                print('+',gram,chars[i+n])
            countcount(forth_freedoms,gram,chars[length-1])
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
test_counters1 = counters_init(2)
assert str(test_counters1) == '([{}, {}], [{}, {}], [{}, {}])'

grams_count_with_char_freedoms(test_counters1[0],test_counters1[1],test_counters1[2],list("abaxb"),1)
grams_count_with_char_freedoms(test_counters1[0],test_counters1[1],test_counters1[2],list("abaxb"),2)
assert str([merge_dicts(d) for d in test_counters1]) == "[{'a': 2, 'b': 2, 'x': 1, 'ab': 1, 'ba': 1, 'ax': 1, 'xb': 1}, {'a': {'b': 2}, 'b': {'b': 1}, 'x': {'b': 1}, 'ab': {'b': 1}, 'ba': {'b': 1}, 'ax': {'b': 1}}, {'b': {'a': 1, 'x': 1}, 'a': {'b': 1}, 'x': {'a': 1}, 'ba': {'a': 1}, 'ax': {'b': 1}, 'xb': {'a': 1}}]"
assert str(model_grams_count_with_char_freedoms(["abaxb"],2)) == "[{'a': 2, 'b': 2, 'x': 1, 'ab': 1, 'ba': 1, 'ax': 1, 'xb': 1}, {'a': {'b': 2}, 'b': {'b': 1}, 'x': {'b': 1}, 'ab': {'b': 1}, 'ba': {'b': 1}, 'ax': {'b': 1}}, {'b': {'a': 1, 'x': 1}, 'a': {'b': 1}, 'x': {'a': 1}, 'ba': {'a': 1}, 'ax': {'b': 1}, 'xb': {'a': 1}}]"
assert str(model_grams_count_with_char_freedoms(["abaxb","abaxb"],2)) == "[{'a': 4, 'b': 4, 'x': 2, 'ab': 2, 'ba': 2, 'ax': 2, 'xb': 2}, {'a': {'b': 4}, 'b': {'b': 2}, 'x': {'b': 2}, 'ab': {'b': 2}, 'ba': {'b': 2}, 'ax': {'b': 2}}, {'b': {'a': 2, 'x': 2}, 'a': {'b': 2}, 'x': {'a': 2}, 'ba': {'a': 2}, 'ax': {'b': 2}, 'xb': {'a': 2}}]"


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
    for i in range(1,length-1):
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
            print("Forw {}-{}:'{}':{}=>{}".format(start,i,forw_gram,sym,forw_freedom))
            print("Back {}-{}:'{}':{}=>{}".format(i,end,back_gram,sym,back_freedom))
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
