
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
    if arg in dic:
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

