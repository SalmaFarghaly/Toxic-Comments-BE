from itertools import cycle

def list_padding(lst , maxlen):
    lstNew = lst.copy()
    myiter = cycle(lstNew)
    for _ in range(maxlen):
        lstNew.append(next(myiter))
    return lstNew

def custom_padding(lst, maxlength=389):
    new =[]
    for  i in  range(len(lst)):
        length =  maxlength - len(lst[i])
        my_padding = list_padding(lst[i] , length)
        new.append( my_padding)
    return new 

