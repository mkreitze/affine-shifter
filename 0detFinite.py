# Idea; enumerate all invertible affine matricies for a given alphabet size
# Write all enumerated values as a sequence and check the OEIS

import numpy as np
import itertools as it

def numberToBase(n, b):
    if n == 0:
        return [0]
    digits = []
    while n:
        digits.append(int(n % b))
        n //= b
    return digits[::-1]



def makeConstConds(alphabet = [0,1]):
    windowSize = 2
    windows = it.product(alphabet,repeat = windowSize)
    validWindows = []
    for window in windows:
        if (window[0] != 0) or (window[1] != 0):
            validWindows.append(window)
    npWindows = np.array(validWindows)
    return(npWindows)

    # detConditions = np.matrix([[0,0,1,1],[0,1,1,1],[1,0,1,1],[1,1,0,0],[1,1,0,1],[1,1,1,0],]) # listed in binary smallest to largest to make pattern apparent later
    # consistConditions = np.matrix([[0,1],[1,0],[1,1]])


# def findFiniteInverts()

# findFiniteInverts(alphabet = [0,1],windowSize = "2x2")

# find R^a = I = D^b
def produceDeBT():
    return()