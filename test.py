import sys
sys.setrecursionlimit(1000000) #例如这里设置为一百万
def power_set(lst):
    if lst == []:
        return [[]]
    rest = power_set(lst[1:])
    result = []
    for item in rest:
        result.append(item)
        result.append([lst[0]] + item)
    return result



print(power_set([1, 2, 3]))
