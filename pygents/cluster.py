import math
from copy import deepcopy as dcp
from pygents.util import dictcount, countcount

#https://stackoverflow.com/questions/41827983/right-way-to-calculate-the-cosine-similarity-of-two-word-frequency-dictionaries
#https://realpython.com/python-counter/

#from scipy.spatial.distance import cosine
#from sklearn.metrics.pairwise import cosine_similarity


#https://github.com/aigents/aigents-java/blob/master/src/main/java/net/webstructor/data/Miner.java#L157
#public double crossOverlap(Linker that, Linker other)
#overlap measure
#https://en.wikipedia.org/wiki/Jaccard_index
def overlap_dic_parts(dic1,dic2):
    numerator = 0
    dena = 0
    for key1,val1 in dic1.items():
        numerator += min(val1,dic2.get(key1,0.0))
        dena += max(val1,val1)
    denb = 0
    for val2 in dic2.values():
        denb += val2
    return numerator, dena, denb
    
def overlap_dic(dic1,dic2):
    numerator, dena, denb = overlap_dic_parts(dic1,dic2)
    return numerator/math.sqrt(dena*denb) if numerator != 0 else 0.0
x0 = {'c':3,'a':1,'b':2}
x1 = {'a':1,'b':2,'c':3}
x2 = {'a':1,'b':2,'c':0}
x3 = {'a':1,'b':2,'d':3}
x4 = {'a':1,'e':2,'d':3}
x5 = {'a':999,'e':2,'d':3}
x6 = {'a':0.1,'e':2,'d':3}
x7 = {'f':1,'e':2,'d':3}
assert str(overlap_dic(x0,x1)) == "1.0"
assert str(overlap_dic(x0,x2)) == "0.7071067811865476"
assert str(overlap_dic(x1,x2)) == "0.7071067811865476"
assert str(overlap_dic(x1,x3)) == "0.5"
assert str(overlap_dic(x1,x4)) == "0.16666666666666666"
assert str(overlap_dic(x1,x5)) == "0.012884201800751829"
assert str(overlap_dic(x1,x6)) == "0.018077538151554683"
assert str(overlap_dic(x1,x7)) == "0.0"

def overlap_dic2(dica1,dicb1,dica2,dicb2):
    numerator1, dena1, denb1 = overlap_dic_parts(dica1,dicb1)
    numerator2, dena2, denb2 = overlap_dic_parts(dica2,dicb2)
    return (numerator1+numerator2)/math.sqrt((dena1+dena2)*(denb1+denb2)) if numerator1 != 0 or numerator2 != 0 else 0.0


#cosine distance measure
#https://en.wikipedia.org/wiki/Cosine_similarity
def cosine_dic_parts(dic1,dic2):
    numerator = 0
    dena = 0
    for key1,val1 in dic1.items():
        numerator += val1*dic2.get(key1,0.0)
        dena += val1*val1
    denb = 0
    for val2 in dic2.values():
        denb += val2*val2
    return numerator, dena, denb

def cosine_dic(dic1,dic2):
    numerator, dena, denb = cosine_dic_parts(dic1,dic2)
    return numerator/math.sqrt(dena*denb) if numerator != 0 else 0.0
x0 = {'c':3,'a':1,'b':2}
x1 = {'a':1,'b':2,'c':3}
x2 = {'a':1,'b':2,'c':0}
x3 = {'a':1,'b':2,'d':3}
x4 = {'a':1,'e':2,'d':3}
x5 = {'a':999,'e':2,'d':3}
x6 = {'a':0.1,'e':2,'d':3}
x7 = {'f':1,'e':2,'d':3}
assert str(cosine_dic(x0,x1)) == "1.0"
assert str(cosine_dic(x0,x2)) == "0.5976143046671968"
assert str(cosine_dic(x1,x2)) == "0.5976143046671968"
assert str(cosine_dic(x1,x3)) == "0.35714285714285715"
assert str(cosine_dic(x1,x4)) == "0.07142857142857142"
assert str(cosine_dic(x1,x5)) == "0.26725950125174264"
assert str(cosine_dic(x1,x6)) == "0.007409643851431125"
assert str(cosine_dic(x1,x7)) == "0.0"

# computes cosine distance based on 2 dicts corresponding to 
# two vectors in complementary two segments of bi-segment vector space
def cosine_dic2(dica1,dicb1,dica2,dicb2):
    numerator1, dena1, denb1 = cosine_dic_parts(dica1,dicb1)
    numerator2, dena2, denb2 = cosine_dic_parts(dica2,dicb2)
    return (numerator1+numerator2)/math.sqrt((dena1+dena2)*(denb1+denb2)) if numerator1 != 0 or numerator2 != 0 else 0.0
assert str(cosine_dic2(x0,x1,{},{})) == "1.0"
assert str(cosine_dic2(x0,x2,{},{})) == "0.5976143046671968"
assert str(cosine_dic2(x1,x2,{},{})) == "0.5976143046671968"
assert str(cosine_dic2(x1,x3,{},{})) == "0.35714285714285715"
assert str(cosine_dic2(x1,x4,{},{})) == "0.07142857142857142"
assert str(cosine_dic2(x1,x5,{},{})) == "0.26725950125174264"
assert str(cosine_dic2(x1,x6,{},{})) == "0.007409643851431125"
assert str(cosine_dic2(x1,x7,{},{})) == "0.0"
assert str(cosine_dic2(x0,x1,x0,x1)) == "1.0"
assert str(cosine_dic2(x1,x7,x1,x7)) == "0.0"

def compute_similiarities(model,arity=1,simfunc2=cosine_dic2,debug=False):
    lst = []
    done = set()
    for a in model[0]:
        if len(a) == arity:
            a1 = model[1][a]
            a2 = model[2][a]
            for b in model[0]:
                if a != b and len(b) == arity and not (b,a) in done:
                    b1 = model[1][b]
                    b2 = model[2][b]
                    s = simfunc2(a1,b1,a2,b2)
                    done.add((a,b))
                    lst.append( (a,b,s) if a <= b else (b,a,s) )
            if debug:
                print(a)
    return lst


def compute_similiarities_from_dict(dic,simfunc2=cosine_dic2,debug=False):
    lst = []
    done = set()
    for a in dic:
            a1 = dic[a][0]
            a2 = dic[a][1]
            for b in dic:
                if a != b and not (b,a) in done:
                    b1 = dic[b][0]
                    b2 = dic[b][1]
                    s = simfunc2(a1,b1,a2,b2)
                    done.add((a,b))
                    lst.append( (a,b,s) if a <= b else (b,a,s) )
            if debug:
                print(a)
    return lst

def model_to_dict(model,arity=1,debug=False):
    copy = {}
    for a in model[0]:
        if len(a) == arity:
            copy[a] = (model[1][a] if a in model[1] else {}, model[2][a] if a in model[2] else {})
    return copy

def dict_merge(a,b):
    c = dcp(a)
    for key in b:
        if key in c:
            c[key] = c[key] + b[key]
        else:
            c[key] = b[key]
    return c
assert str(dict_merge({'a':0.2,'b':0.1},{'c':0.2,'b':0.1})) == "{'a': 0.2, 'b': 0.2, 'c': 0.2}"         

def join_letters(a,b):
    return "".join(sorted(list(a)+list(b)))
assert str(join_letters("1.2","zba")) == ".12abz"
    
def do_cluster(model,dic4tree=None,simfunc2=cosine_dic2,debug = False):
    copy = model_to_dict(model)
    if debug:
        print(len(copy))
    n = 0
    while True:
        simlst = compute_similiarities_from_dict(copy,simfunc2=simfunc2)
        simlst.sort(key=lambda tup: tup[2], reverse=False) # sort to end so we can be removing from the end
        length = len(simlst)
        if length == 0:
            break # root
        top = simlst[length - 1]
        merged_name = join_letters(top[0],top[1])
        if not dic4tree is None:
            dic4tree[top[0]] = merged_name
            dic4tree[top[1]] = merged_name
        if debug:
            print(n,len(copy),length,top[0],'+',top[1],'=>',top[2])
        copy[ merged_name ] = ( dict_merge(copy[top[0]][0],copy[top[1]][0]), dict_merge(copy[top[0]][1],copy[top[1]][1]) )
        del copy[top[0]]
        del copy[top[1]]
        n += 1
    if debug:
        print(len(copy))


def do_cluster_threshold(model,dic4tree=None,simfunc2=cosine_dic2,thresholds=[0.95,0.9,0.85,0.8,0.75,0.7,0.65,0.6,0.55,0.5,0.45,0.4],debug = False):
    copy = model_to_dict(model)
    if debug:
        print('start count',len(copy))
    for threshold in thresholds:
        if debug:
            print('threshold',threshold)
        while True:
            simlst = compute_similiarities_from_dict(copy,simfunc2=simfunc2)
            simlst.sort(key=lambda tup: tup[2],reverse=True)
            length = len(simlst)
            if length == 0:
                break # root
            merged_count = 0
            merged_set = set()
            for top in simlst:
                if top[2] < threshold: # stop merging on threshold
                    break
                if top[0] in merged_set or top[1] in merged_set: # skip those merged on current round
                    continue
                merged_name = join_letters(top[0],top[1])
                if not dic4tree is None:
                    dic4tree[top[0]] = merged_name
                    dic4tree[top[1]] = merged_name
                if debug:
                    print(len(copy),length,top[0],'+',top[1],'=>',top[2])
                copy[ merged_name ] = ( dict_merge(copy[top[0]][0],copy[top[1]][0]), dict_merge(copy[top[0]][1],copy[top[1]][1]) )
                del copy[top[0]]
                del copy[top[1]]
                merged_set.add(top[0])
                merged_set.add(top[1])
                merged_count += 1
            if merged_count == 0:
                break
    if debug:
        print('end count',len(copy))


def invert_model(model,maxn=2):
    index = {}
    forward = {}
    backward = {}
    for ngram in model[1]:
        if len(ngram) <= maxn: 
            model1ngram = model[1][ngram]
            for e in model1ngram:
                dictcount(index,e)
                countcount(backward,e,ngram,model1ngram[e])
    for ngram in model[2]:
        if len(ngram) <= maxn: 
            model2ngram = model[2][ngram]
            for e in model2ngram:
                dictcount(index,e)
                countcount(forward,e,ngram,model2ngram[e])
    #TODO denominate all index by 2 !?
    return index, forward, backward  
assert str(invert_model([{'a':100,'b':10},
              {'a':{'b':10},'b':{'a':100,'c':1}},
              {'a':{'b':20},'b':{'a':200,'c':2}}
             ])) == "({'b': 2, 'a': 2, 'c': 2}, {'b': {'a': 20}, 'a': {'b': 200}, 'c': {'b': 2}}, {'b': {'a': 10}, 'a': {'b': 100}, 'c': {'b': 1}})"
assert str(invert_model([{'a':100,'b':10,'dd':1},
              {'a':{'b':10},'b':{'a':100,'c':1},'dd':{'a':1,'c':1}},
              {'a':{'b':20},'b':{'a':200,'c':2},'dd':{'a':1,'c':1}}
             ],maxn=1)) == "({'b': 2, 'a': 2, 'c': 2}, {'b': {'a': 20}, 'a': {'b': 200}, 'c': {'b': 2}}, {'b': {'a': 10}, 'a': {'b': 100}, 'c': {'b': 1}})"

