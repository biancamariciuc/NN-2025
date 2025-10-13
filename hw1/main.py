#ex
import pathlib
import math
import copy

def load_system(path: pathlib.Path) -> tuple[list[list[float]], list[float]]:
    matA = []
    matB = []
    file = open(path)
    for line in file:
        left, right = line.split("=")
        right = float(right.strip())
        elements = left.strip().split()
        sign = 1
        coefX = coefY = coefZ = 0.0
        for element in elements:
            if element == '-':
                sign = -1
            elif element == '+':
                sign = 1
            else:
                if 'x' in element:
                    element = element.replace('x', '')
                    if element == '':
                        coef = 1.0
                    elif element == '-':
                        coef = -1.0
                    else:
                        coef = float(element)
                    coefX = coef * sign
                elif 'y' in element:
                    element = element.replace('y', '')
                    if element == '':
                        coef = 1.0
                    elif element == '-':
                        coef = -1.0
                    else:
                        coef = float(element)
                    coefY = coef * sign
                elif 'z' in element:
                    element = element.replace('z', '')
                    if element == '':
                        coef = 1.0
                    elif element == '-':
                        coef = -1.0
                    else:
                        coef = float(element)
                    coefZ = coef * sign

        matA.append([coefX, coefY, coefZ])
        matB.append(right)
    return matA, matB

A, B = load_system(pathlib.Path("systems.txt"))
print(f"{A=} {B=}")

def determinant(matrix: list[list[float]]) -> float:
    if len(matrix) == 2:
        det = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    elif len(matrix) == 3:
        expression1 = matrix[0][0] * (matrix[1][1] * matrix[2][2]  - matrix[1][2] * matrix[2][1])
        expression2 = matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        exxpression3 = matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
        det = expression1 - expression2 + exxpression3
    return det

print(f"{determinant(A)=}")

def trace(matrix: list[list[float]]) -> float:
    tr = 0
    for i in range(len(matrix)):
        tr += matrix[i][i]
    return tr
print(f"{trace(A)=}")

def norm(vector: list[float]) -> float:
    euclidNorm = 0
    for i in range(len(vector)):
        euclidNorm += vector[i] * vector[i]
    return math.sqrt(euclidNorm)

print(f"{norm(B)=}")

# firstElement = matrix[0][0]*vector[0] + matrix[0][1]*vector[1] + matrix[0][2]*vector[2]
def multiply(matrix: list[list[float]], vector: list[float]) -> list[float]:

    resultMatrix = []
    for i in range(len(matrix)):
        element = 0
        for j in range(len(matrix)):
            element += matrix[i][j] * vector[j]
        resultMatrix.append(element)

    return resultMatrix

print(f"{multiply(A, B)=}")

def solve_cramer(matrix: list[list[float]], vector: list[float]) -> list[float]:
    matrixX = copy.deepcopy(matrix)
    matrixY = copy.deepcopy(matrix)
    matrixZ = copy.deepcopy(matrix)

    for i in range(3):
        matrixX[i][0] = vector[i]
        matrixY[i][1] = vector[i]
        matrixZ[i][2] = vector[i]

    detA = determinant(matrix)
    if detA == 0:
        return []

    detAx = determinant(matrixX)
    detAy = determinant(matrixY)
    detAz = determinant(matrixZ)

    x = detAx/detA
    y = detAy/detA
    z = detAz/detA

    return [x, y, z]

print(f"{solve_cramer(A, B)=}")

def minor(matrix: list[list[float]], i: int, j: int) -> list[list[float]]:
    minorMatrix = []

    for ii in range (len(matrix)):
        if ii == i:
            continue
        row = []
        for jj in range (len(matrix)):
            if jj == j:
                continue
                # print(matrix[ii][jj])
            row.append(matrix[ii][jj])
        minorMatrix.append(row)

    return minorMatrix

# print(f"{minor(A, 0, 2)=}")

def cofactor(matrix: list[list[float]]) -> list[list[float]]:
    cofactorMatrix = []

    for i in range(len(matrix)):
        row = []
        for j in range(len(matrix)):
            sign = (-1) ** (i + j)
            minorMatrix = minor(matrix, i, j)
            detMinor = determinant(minorMatrix)
            row.append(sign * detMinor)
        cofactorMatrix.append(row)

    return cofactorMatrix


def adjoint(matrix: list[list[float]]) -> list[list[float]]:
    cofactorMatrix = cofactor(matrix)
    for i in range(len(cofactorMatrix)):
        for j in range(i+1, len(cofactorMatrix)):
            cofactorMatrix[i][j], cofactorMatrix[j][i] = cofactorMatrix[j][i], cofactorMatrix[i][j]
    return cofactorMatrix

def solve(matrix: list[list[float]], vector: list[float]) -> list[float]:
    detA = determinant(matrix)
    if detA == 0:
        return []

    inverseMatrix = adjoint(matrix)

    for i in range(len(inverseMatrix)):
        for j in range(len(inverseMatrix)):
            inverseMatrix[i][j] /= detA;

    result = multiply(inverseMatrix, vector)
    return result


print(f"{solve(A, B)=}")