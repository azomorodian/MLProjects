import numpy as np
a = np.arange(12).reshape((3, 4))
print(a)

print(np.count_nonzero(a))

print(np.count_nonzero(a<4))

print(np.sum(a % 2 == 0))

print(np.count_nonzero(a < 4, axis=0))
print(np.count_nonzero(a < 4, axis=1))

print(np.count_nonzero(a < 4, axis=0, keepdims=True))
print(np.count_nonzero(a < 4, axis=1, keepdims=True))

a = np.arange(12).reshape((3, 4))
print(a)

print(np.any(a < 4))
print(np.any(a > 100))

print(np.any(a < 4, axis=0))
print(np.any(a < 4, axis=1))

print(np.all(a < 4))
print(np.all(a < 100))

print(np.all(a < 4, axis=0))
print(np.all(a < 4, axis=1))

#Multiple conditions
a = np.arange(12).reshape((3, 4))
print(a)
# [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]]

print((a < 4) | (a % 2 == 0))
# [[ True  True  True  True]
#  [ True False  True False]
#  [ True False  True False]]

print(np.count_nonzero((a < 4) | (a % 2 == 0)))
# 8

print(np.count_nonzero((a < 4) | (a % 2 == 0), axis=0))
# [3 1 3 1]

print(np.count_nonzero((a < 4) | (a % 2 == 0), axis=1))
# [4 2 2]

print(np.count_nonzero((a < 4) | (a % 2 == 0), axis=0, keepdims=True))

a_nan = np.genfromtxt('sample_nan.csv', delimiter=',')
print(("Matrix : {}\nShape : {} ").format(a_nan,np.shape(a_nan)))

#Since comparing NaN with NaN always returns False
print(np.nan == np.nan)
# False
print(np.isnan(a_nan))

print(np.count_nonzero(np.isnan(a_nan)))
# 3
print(np.count_nonzero(np.isnan(a_nan), axis=0))
# [0 1 2 0]

print(np.count_nonzero(np.isnan(a_nan), axis=1))
# [1 2 0]

print(~np.isnan(a_nan))
# [[ True  True False  True]
#  [ True False False  True]
#  [ True  True  True  True]]


a_inf = np.array([-np.inf, 0, np.inf])
print(a_inf)
# [-inf   0.  inf]

print(np.isinf(a_inf))
# [ True False  True]

print(np.isposinf(a_inf))
# [False False  True]

print(np.isneginf(a_inf))
# [ True False False]

print(a_inf == np.inf)
# [False False  True]

print(a_inf == -np.inf)
# [ True False False]

print(np.count_nonzero(np.isinf(a_inf)))
# 2

print(np.count_nonzero(np.isposinf(a_inf)))
# 1

print(np.count_nonzero(np.isneginf(a_inf)))
# 1





