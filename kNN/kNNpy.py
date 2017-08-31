from numpy import *

matrix = mat(random.rand(4, 4))
print matrix
reversedMatrix = matrix.I
print reversedMatrix
print (matrix * reversedMatrix)

