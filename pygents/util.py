
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
