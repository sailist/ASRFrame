num = ['零', '一', '二', '三', '四', '五', '六', '七', '八', '九']
kin = ['十', '百', '千', '万', '零']
import time


def sadd(x):
    x.reverse()
    if len(x) >= 2:
        x.insert(1, kin[0])
        if len(x) >= 4:
            x.insert(3, kin[1])
            if len(x) >= 6:
                x.insert(5, kin[2])
                if len(x) >= 8:
                    x.insert(7, kin[3])
                    if len(x) >= 10:
                        x.insert(9, kin[0])
                        if len(x) >= 12:
                            x.insert(11, kin[1])

    x = fw(x)
    x = d1(x)
    x = d2(x)
    x = dl(x)
    return x

def convert(i,pure_number = False):
    raw = i
    try:
        if pure_number:
            i = list(str(i))
            for j in i:
                i[(i.index(j))] = num[int(j)]
            i = "".join(i)
        else:
            i = list(str(i))
            for j in i:
                i[(i.index(j))] = num[int(j)]
            i = sadd(i)
    except:
        if pure_number:
            return ""
        return convert(raw,True)
    return i


def rankis():
    rank = []
    for i in range(10086):
        if True:
            i = list(str(i))
            for j in i:
                i[(i.index(j))] = num[int(j)]
            rank.append("".join(i))
            continue

        i = list(str(i))
        for j in i:
            i[(i.index(j))] = num[int(j)]
        i = sadd(i)
        rank.append(i)
    return rank


def d1(x):
    if '零' in x:
        a = x.index('零')
        if a == 0:
            del x[0]
            d1(x)
        else:
            if x[a + 2] in ['十', '百', '千', '万', '零']:
                if x[a + 1] != '万':
                    del x[a + 1]
                    d1(x)
    return x


def d2(x):
    try:
        a = x.index('零')
        if x[a - 1] in ['十', '百', '千', '零']:
            del x[a - 1]
            d2(x[a + 1])
    except:
        pass
    return x


def fw(x):
    if len(x) >= 9:
        if x[8] == '零':
            del x[8]
    return x


def dl(x):
    try:
        if x[0] == '零':
            del x[0]
            d1(x)
    except:
        pass
    x.reverse()
    x = ''.join(x)
    return x

if __name__ == "__main__":
    print(convert("0",True))
