import numpy as np
import pickle
import math
from os.path import join
from scipy.stats import entropy

def vector_proximity(avector,bvector,threshold):
    for a,b in zip(avector,bvector):
        ab = a*b
        if ab < 0: # check if sign is different
            return False
        if ab != 0:
            if abs((a-b)/math.sqrt(ab)) > threshold:
                #print(a,b,abs((a-b)/math.sqrt(ab)))
                return False
    return True
assert not vector_proximity((10,-10,10),(10,10,10),0.1)
assert vector_proximity((10,-10,10),(10,-10,10),0.1)
assert vector_proximity((10,-10,10),(10,-10,10),0.1)
assert not vector_proximity((10,-10,10),(19,-19,19),0.1)

def dict_update(target,source):
    for key in source:
        if not key in target:
            target[key] = source[key]
        else:
            target_value = target[key]
            source_value = source[key]
            if isinstance(source_value, dict):
                assert isinstance(target_value, dict)
                dict_update(target_value,source_value)
            else:
                assert (type(target_value) == int or float) and (type(source_value) == int or float)
                target[key]= target_value + source_value
    return target
assert str(dict_update({'a':1},{'a':1,'b':20})) == "{'a': 2, 'b': 20}"
assert str(dict_update({'a':1,'c':{'x':100}},{'a':1,'b':20,'c':{'x':300,'y':4000},'z':{'x':50000}})) == "{'a': 2, 'c': {'x': 400, 'y': 4000}, 'b': 20, 'z': {'x': 50000}}"


def listofpairs_compress_with_loss(lst,threshold=0.01):
    maxval = None
    for i in lst:
        if maxval is None or maxval < i[1]:
            maxval = i[1]
    newlist = []
    minval = maxval * threshold
    for i in lst:
        if i[1] >= minval:
            newlist.append(i)
    return newlist
assert str(listofpairs_compress_with_loss([('a',1000),('b',100),('c',10),('d',1)])) == "[('a', 1000), ('b', 100), ('c', 10)]"


def dict_compress_with_loss(dic,threshold=0.01):
    maxval = None
    for d in dic:
        v=dic[d]
        if isinstance(v,dict):
            dict_compress_with_loss(v,threshold) # recursion
        else:
            assert (type(v) == int or float)
            if maxval is None or maxval < v:
                maxval = v
    if maxval is not None:
        todo = []
        minval = maxval * threshold
        for d in dic:
            if dic[d] < minval:
                 todo.append(d)
        for d in todo:
            del dic[d]
    return dic
assert str(dict_compress_with_loss({'a':1000,'b':10,'c':1})) == "{'a': 1000, 'b': 10}"
assert str(dict_compress_with_loss({'x':{'a':1000,'b':10,'c':1},'y':{'m':2000,'n':20,'o':2}})) == "{'x': {'a': 1000, 'b': 10}, 'y': {'m': 2000, 'n': 20}}"


def dict_of_dicts_compress_by_threshold_unscaled(dict_of_dicts, inclusion_threshold, rescale = False):
    #print('compact_dict_of_dicts_by_threshold',inclusion_threshold)
    filtered_dict_of_dicts = {}
    for label, ngram_dict in dict_of_dicts.items():
        # Find the maximum metric value for the current label
        max_value = max(ngram_dict.values()) if ngram_dict else 0
        threshold_value = max_value * (inclusion_threshold / 100)

        # Filter n-grams that meet or exceed the threshold value
        filtered_dict_of_dicts[label] = {
            ngram: metric for ngram, metric in ngram_dict.items() if metric >= threshold_value
        }
    return filtered_dict_of_dicts
assert str(dict_of_dicts_compress_by_threshold_unscaled({'x':{'a':0.2,'b':0.1},'y':{'c':0.4,'b':0.2}},60)) == "{'x': {'a': 0.2}, 'y': {'c': 0.4}}"


def dict_of_dicts_compress_by_threshold(dict_of_dicts, inclusion_threshold, rescale = False):
    #print('compact_dict_of_dicts_by_threshold',inclusion_threshold)
    filtered_dict_of_dicts = {}
    for label, ngram_dict in dict_of_dicts.items():
        # Find the maximum metric value for the current label
        max_value = max(ngram_dict.values()) if ngram_dict else 0
        threshold_value = max_value * (inclusion_threshold / 100)
        factor = 1.0 / max_value if max_value != 0 else 0

        # Filter n-grams that meet or exceed the threshold value
        if rescale:
            filtered_dict_of_dicts[label] = {
                ngram: metric * factor for ngram, metric in ngram_dict.items() if metric >= threshold_value
            } 
        else:
            filtered_dict_of_dicts[label] = {
                ngram: metric for ngram, metric in ngram_dict.items() if metric >= threshold_value
            }
    return filtered_dict_of_dicts
assert str(dict_of_dicts_compress_by_threshold({'x':{'a':0.2,'b':0.1},'y':{'c':0.4,'b':0.2}},60)) == "{'x': {'a': 0.2}, 'y': {'c': 0.4}}"
assert str(dict_of_dicts_compress_by_threshold({'x':{'a':0.2,'b':0.1},'y':{'c':0.4,'b':0.2}},60,rescale=True)) == "{'x': {'a': 1.0}, 'y': {'c': 1.0}}"


def dictdict_div_dict(num, den, default = 1, debug = False):
    res = {}
    for n in num:
        if isinstance(num[n], dict):
            res[n] = dictdict_div_dict(num[n], den)
        else:
            if debug:
                print(n,num[n],den[n],type(num[n]),type(den[n]))
                #k = float(num[n]) / den[n]
            if den[n] == 0:
                res[n] = float(num[n]) / default
            else:
                res[n] = float(num[n]) / den[n]
    return res
n = {"a":{"x":10,"y":15},"b":{"y":30,"z":40}}; d = {"x":20,"y":30,"z":40}; assert str(dictdict_div_dict(n,d))=="{'a': {'x': 0.5, 'y': 0.5}, 'b': {'y': 1.0, 'z': 1.0}}"


def dictdict_mul_dictdict(num, den):
    res = {}
    for n in num:
        if isinstance(num[n], dict):
            #print(num[n], den[n])
            res[n] = dictdict_mul_dictdict(num[n], den[n])
            #print(res[n])
        else:
            res[n] = float(num[n]) * den[n]
    return res
n = {"a":{"x":10,"y":20},"b":{"x":20,"y":40}}; d = {"a":{"x":10,"y":5},"b":{"x":0.5,"y":0.25}}; assert str(dictdict_mul_dictdict(n,d))=="{'a': {'x': 100.0, 'y': 100.0}, 'b': {'x': 10.0, 'y': 10.0}}"


def dict2listsorted(d):
    return [(key, value) for key, value in sorted(d.items())]

def dict_diff(a,b):
    diff = {}
    for key in set(a).union(set(b)):
        if key in a and key in b:
            delta = a[key] - b[key]
            if delta != 0:
                diff[key] = delta
        elif key in a:
            diff[key] = a[key]
        elif key in b: 
            diff[key] = - b[key]
    return diff
assert str(dict2listsorted(dict_diff({'a':1,'b':2,'c':3,'d':1},{'a':1,'b':3,'c':2,'x':1}))) == "[('b', -1), ('c', 1), ('d', 1), ('x', -1)]"

def remove_all(collection,item):
    while item in collection:
        collection.remove(item)

def dictcount(dic,arg,cnt=1):
    if type(arg) == list:
        #print(arg)
        #print(dic)
        for i in arg:
            dictcount(dic,i,cnt)
    elif arg in dic:
        dic[arg] = dic[arg] + cnt
    else:
        dic[arg] = cnt

def merge_two_dicts(x, y):
    """Given two dictionaries, merge them into a new dict as a shallow copy."""
    z = x.copy()
    z.update(y)
    return z

def countcount(dic,arg,subarg,cnt=1):
    if arg in dic:
        subdic = dic[arg]
    else:
        dic[arg] = subdic = {}
    dictcount(subdic,subarg,cnt)

def counters_init(max_n):  
    return [{} for n in range(max_n)], [{} for n in range(max_n)], [{} for n in range(max_n)]

def merge_dicts(dicts):
    merged={}
    for d in dicts:
        merged.update(d)
    return merged

def count_subelements(element):
    count = 0
    if isinstance(element, list) or isinstance(element, tuple):
        for child in element:
            count += count_subelements(child)
    elif isinstance(element, dict):
        for key in element:
            count += count_subelements(element[key])
    else:
        count = 1
    return count 
assert count_subelements(['1',2,[[3,'4',{'x':['5',6],'y':(7,'8')},{'z':{'p':9,'q':['10']}}]]]) == 10

def contains_seq(B,A):
    if type(A) == list:
        A = tuple(A)
    if type(B) == list:
        B = tuple(B)
    return any(A == B[i:len(A) + i] for i in range(len(B) - len(A) + 1))
assert contains_seq(('L', 'R', 'L', 'L', 'L', 'R'),('L', 'R'))


def sqrt_sum_squares_dict_values(dct):
    return math.sqrt(sum(x*x for x in dct.values()))
        
def cosine_similarity(dict_1, dict_2):
    intersecting_keys = list(dict_1.keys() & dict_2.keys())

    List1 = list(dict_1[k] for k in intersecting_keys)
    List2 = list(dict_2[k] for k in intersecting_keys)
    
    similarity = np.dot(List1,List2) / (sqrt_sum_squares_dict_values(dict_1) * sqrt_sum_squares_dict_values(dict_2))
    return round(similarity, 2)

assert cosine_similarity({"a": 1, "b": 2, "c": 3}, {"c": 5, "b": 4, "d": 6}) == 0.7
assert cosine_similarity({"a": 1.0, "b": 0.5, "c": 0.1}, {"a": 1.0, "b": 0.4, "d": 0.1}) == 0.99


# Counting measures 

#https://en.wikipedia.org/wiki/F-score
def dict_precision(ground,guess):
    true_positives = sum([min(guess[key],ground[key]) if key in ground else 0 for key in guess])
    guess_positives = sum(guess.values())
    return true_positives / guess_positives

def dict_recall(ground,guess):
    true_positives = sum([min(guess[key],ground[key]) if key in ground else 0 for key in guess])
    ground_positives = sum(ground.values())
    return true_positives / ground_positives

def list2dict(lst):
    dic = {}
    for l in lst:
        dictcount(dic,l)
    return dic

def listofpairs2dict(lst):
    dic = {}
    for i in lst:
        dictcount(dic,i[0],i[1])
    return dic

def round_str(val,decimals=0):
    if val == 0:
        return '0.'+"".join('0'*decimals)  
    s = str(round(val,decimals))
    point = s.find(".")
    zeros = decimals - (len(s) - point) + 1
    #print(point,len(s),zeros)
    return s + ('0'*zeros)

def calc_f1(ground,guess):
    if isinstance(ground,list):
        ground = list2dict(ground)
    if isinstance(guess,list):
        guess = list2dict(guess)
    true_positives = sum([min(guess[key],ground[key]) if key in ground else 0 for key in guess])
    guess_positives = sum(guess.values())
    ground_positives = sum(ground.values())
    precision = true_positives / guess_positives
    recall = true_positives / ground_positives
    return 2 * precision * recall / (precision + recall) if precision > 0 or recall > 0 else 0 

assert dict_precision({'a':1,'b':1},{'a':1,'c':1}) == 0.5
assert dict_precision({'x':2},{'x':1}) == 1.0
assert dict_precision({'x':2},{'x':4}) == 0.5
assert dict_precision({'a':1,'b':1,'x':1},{'a':1,'c':1,'x':2}) == 0.5
assert dict_recall({'a':1,'b':1},{'a':1,'c':1}) == 0.5
assert dict_recall({'x':2},{'x':1}) == 0.5
assert dict_recall({'x':2},{'x':4}) == 1.0
assert dict_recall({'a':1,'x':1},{'a':1,'c':1,'x':2}) == 1.0
assert dict_recall({'a':1,'b':1,'x':1},{'a':1,'c':1,'x':2}) == 2/3
assert calc_f1({'a':1,'x':1},{'b':1,'y':1}) == 0
assert calc_f1({'a':1,'b':1},{'a':1,'c':1}) == 0.5
assert calc_f1({'a':1,'b':1,'c':1},{'a':1,'b':1}) == 0.8
assert calc_f1({'a':1,'b':1},{'a':1,'b':1,'c':1}) == 0.8
assert calc_f1({'a':1,'b':2},{'a':2,'b':4}) == 2/3
assert calc_f1({'a':2,'b':4},{'a':1,'b':2}) == 2/3
assert str(list2dict(['a','b','a','c','c','c'])) == "{'a': 2, 'b': 1, 'c': 3}"
assert calc_f1(['ab','cd','ef','gh'],['ab','cd','ef','gh']) == 1.0
assert calc_f1(['ab','cd','ef','gh'],['a','b','cd','e','f','gh']) == 0.4
assert calc_f1(['ab','cd','ef','gh'],['abcd','ef','gh']) == 0.5714285714285715
assert calc_f1(['ab','cd','ef','gh'],['abcd','ef']) == 1/3

def calc_diff(ground,guess):
    if isinstance(ground,list):
        ground = list2dict(ground)
    if isinstance(guess,list):
        guess = list2dict(guess)
    return dict_diff(ground,guess)

def list2matrix(lst):
    rows = 0
    cols = 0
    rows_dict = {}
    cols_dict = {}
    # create labels
    for i in lst:
        row = i[0]
        col = i[1]
        if not row in rows_dict:
            rows_dict[row] = rows
            rows += 1
        if not col in cols_dict:
            cols_dict[col] = cols
            cols += 1
    print(rows,cols)
    print(rows_dict)
    print(cols_dict)
    matrix = np.zeros((rows,cols),dtype=float)
    for i in lst:
        row = i[0]
        col = i[1]
        val = i[2]
        matrix[rows_dict[row],cols_dict[col]] = val
    return sorted(set(rows_dict)), sorted(set(rows_dict)), matrix

def list2matrix(lst):
    rows = 0
    cols = 0
    rows_dict = {}
    cols_dict = {}
    rows_list = []
    cols_list = []
    # create labels
    for i in lst:
        row = str(i[0])
        col = str(i[1])
        if not row in rows_dict:
            rows_dict[row] = rows
            rows += 1
            rows_list.append(row)
        if not col in cols_dict:
            cols_dict[col] = cols
            cols += 1
            cols_list.append(col)
    #print(rows,cols)
    #print(rows_dict)
    #print(cols_dict)
    matrix = np.zeros((rows,cols),dtype=float)
    for i in lst:
        row = str(i[0])
        col = str(i[1])
        val = i[2]
        matrix[rows_dict[row],cols_dict[col]] = val
    return rows_list, cols_list, matrix


def context_save_load(context,context_name,folder='data/temp/'):
    ##https://stackoverflow.com/questions/12544056/how-do-i-get-the-current-ipython-jupyter-notebook-name
    pickle_name = join(folder,context_name)
    if context is None:
        context = pickle.load(open(pickle_name, 'rb'))
    else:
        pickle.dump(context, open(pickle_name, 'wb'), pickle.HIGHEST_PROTOCOL)
    return context


def evaluate_entropy(tokenized_texts,counts=None):
    """
    Normalized entropy 
    """
    lexicon = {}
    tokens_count = 0
    for i in range(len(tokenized_texts)):
        tokenized_text = tokenized_texts[i]
        count = 1 if counts is None else counts[i]
        tokens_count += len(tokenized_text) * count
        for token in tokenized_text:
            dictcount(lexicon,token,count)
    distribution = [lexicon[token]/tokens_count for token in lexicon]
    e = entropy(distribution, base=2)
    k = len(lexicon)
    if k > 2:
        e /= math.log2(k)
    return e
#print(evaluate_entropy([["aaaabbbbaaaabbbb"]]))
#print(evaluate_entropy([["aaaabbbb"],["aaaabbbb"]]))
#print(evaluate_entropy([["aaaa"],["bbbb"],["aaaa"],["bbbb"]]))
#print(evaluate_entropy([["aa"],["aa"],["bb"],["bb"],["aa"],["aa"],["bb"],["bb"]]))
#print(evaluate_entropy([["a"],["a"],["a"],["a"],["b"],["b"],["b"],["b"],["a"],["a"],["a"],["a"],["b"],["b"],["b"],["b"]]))
assert str(evaluate_entropy([["aaaabbbbaaaabbbb"]])) == "0.0"
assert str(evaluate_entropy([["aaaabbbb"],["aaaabbbb"]])) == "0.0"
assert str(evaluate_entropy([["aaaa"],["bbbb"],["aaaa"],["bbbb"]])) == "1.0"
assert str(evaluate_entropy([["aa"],["aa"],["bb"],["bb"],["aa"],["aa"],["bb"],["bb"]])) == "1.0"
assert str(evaluate_entropy([["a"],["a"],["a"],["a"],["b"],["b"],["b"],["b"],["a"],["a"],["a"],["a"],["b"],["b"],["b"],["b"]])) == "1.0"


def evaluate_anti_entropy(tokenized_texts,counts=None):
    """
    Normalized anti-entropy 
    """
    return 1.0 - evaluate_entropy(tokenized_texts,counts=counts)


def evaluate_compression(texts,tokenized_texts,counts=None):
    """
    Coefficient of compression 
    """
    text_len = 0
    tokenized_text_len = 0
    tokens_count = 0
    lexicon = {}
    for i in range(len(texts)):
        text_len += len(texts[i]) * (1 if counts is None else counts[i])
    for i in range(len(tokenized_texts)):
        tokenized_text = tokenized_texts[i]
        count = (1 if counts is None else counts[i])
        tokens_count += len(tokenized_text) * count
        for token in tokenized_text:
            tokenized_text_len += len(token) * count # just sanity checking
            dictcount(lexicon,token,count)
    tokens_len = 0
    for token in lexicon:
        tokens_len += len(token)
    assert text_len == tokenized_text_len
    return 1.0 - ((tokens_len + tokens_count) / text_len)
#print(evaluate_compression(["aaaabbbbaaaabbbb"],[["aaaabbbbaaaabbbb"]]))
#print(evaluate_compression(["aaaabbbbaaaabbbb"],[["aaaabbbb"],["aaaabbbb"]]))
#print(evaluate_compression(["aaaabbbbaaaabbbb"],[["aaaa"],["bbbb"],["aaaa"],["bbbb"]]))
#print(evaluate_compression(["aaaabbbbaaaabbbb"],[["aa"],["aa"],["bb"],["bb"],["aa"],["aa"],["bb"],["bb"]]))
#print(evaluate_compression(["aaaabbbbaaaabbbb"],[["a"],["a"],["a"],["a"],["b"],["b"],["b"],["b"],["a"],["a"],["a"],["a"],["b"],["b"],["b"],["b"]]))
assert str(evaluate_compression(["aaaabbbbaaaabbbb"],[["aaaabbbbaaaabbbb"]])) == "-0.0625"
assert str(evaluate_compression(["aaaabbbbaaaabbbb"],[["aaaabbbb"],["aaaabbbb"]])) == "0.375"
assert str(evaluate_compression(["aaaabbbbaaaabbbb"],[["aaaa"],["bbbb"],["aaaa"],["bbbb"]])) == "0.25"
assert str(evaluate_compression(["aaaabbbbaaaabbbb"],[["aa"],["aa"],["bb"],["bb"],["aa"],["aa"],["bb"],["bb"]])) == "0.25"
assert str(evaluate_compression(["aaaabbbbaaaabbbb"],[["a"],["a"],["a"],["a"],["b"],["b"],["b"],["b"],["a"],["a"],["a"],["a"],["b"],["b"],["b"],["b"]])) == "-0.125"


def agg_min_max_avg_mpe(runs):
    max_v = max(runs)
    min_v = min(runs)
    avg_v = sum(runs)/len(runs)
    # https://en.wikipedia.org/wiki/Mean_absolute_error
    mpe_v = sum([abs(v-avg_v) for v in runs])/len(runs)/avg_v*100 if avg_v > 0 else 0
    return min_v, max_v, avg_v, mpe_v

