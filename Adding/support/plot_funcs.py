# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from datetime import datetime
        
def printer(*args, Note='(づ￣ ³￣)づ'):
    print(f'Note {datetime.now().time()}', args)
    
def info(*args):
    print(f'ಠ_ರೃ {datetime.now().time()}',args)
    
def shrug(*args):
    print(f'¯\_(ツ)_/¯ {datetime.now().time()}',args)
    
def debug(*args):
    print(f'(◣_◢) {datetime.now().time()}',args)