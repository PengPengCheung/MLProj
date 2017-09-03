# coding=utf-8

import re

def textParse(bigStr):
    listOfTokens = re.split(r'\W*', bigStr)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]



if __name__ == '__main__':
    mySent='This book is the best book on Python or M.L. I have ever laid eyes upon.'
    print textParse(mySent)
