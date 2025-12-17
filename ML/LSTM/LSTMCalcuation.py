import math
def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def tanh(x):
    return (math.exp(x)-math.exp(-x))/(math.exp(x)+math.exp(-x))

print(tanh(-10))

def lstmCell(ltm,stm,inp):
    #first stage forget percentage
    iw1 = 1.63
    stw1 = 2.7
    b1 = 1.62
    il = iw1*inp + stw1*stm + b1
    out1 = sigmoid(il)*ltm
    print("out1 = " , out1)

    #Potential long term memory
    stw2 = 1.41
    iw2 = 0.94
    b2 = -0.32
    ipltm = iw2*inp + stw2*stm + b2
    out2 = tanh(ipltm)
    print("out2 =",out2)

    #Potential memory to remember
    stw3 = 2
    iw3 = 1.65
    b3 = 0.62
    pmtr = iw3*inp + stw3*stm + b3
    out3 = sigmoid(pmtr)
    print("out3 = ",out3)

    out4 = out2 * out3
    print("out4 = ",out4)

    out5 = out1 + out4
    print("out5 =",out5)

    #Potential short term memorey to remember
    out6 = tanh(out5)
    print("out6 =",out6)
    #Potential Memorey to remember
    iw4 = -0.19
    stw4 = 4.38
    b4 = 0.59
    out7 = sigmoid(iw4*inp + stw4*stm + b4)
    print("out7 =",out7)
    newShortTermMem  = out7 * out6
    newLongTermMem = out5
    return newLongTermMem,newShortTermMem
d1l,d1s = lstmCell(0,0,1)
print(d1l,d1s)
d2l,d2s = lstmCell(d1l,d1s,0.5)
print(d2l,d2s)
d3l,d3s = lstmCell(d2l,d2s,0.25)
print(d3l,d3s)
d4l,d4s = lstmCell(d3l,d3s,1)
print(d4l,d4s)






