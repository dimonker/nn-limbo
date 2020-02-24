import numpy as np

sz = 34
printlogs = True

def m(name):
    if printlogs:
        print(name.center(sz * 2))

def me(name):
    if printlogs:
        print('END {} END'.format(name).center(sz * 2))

def pr(text, val):
    if printlogs:
        label = ' - {} ='.format(text.ljust(sz))
        if isinstance(val, np.ndarray):
            print(label)
            print(val)
        else:
            print(label, val)
        