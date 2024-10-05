from copy import deepcopy
from fractions import Fraction
from tabulate import tabulate
from random import randrange


class MatrixInitError(Exception):
    def __init__(self):
        self.message = "Invalid size of matrix"


class MatrixAdditionError(Exception):
    def __init__(self):
        self.message = "Different sizes"


class MatrixMultiplicationError(Exception):
    def __init__(self):
        self.message = "Invalid size of matrixes"


class MatrixScalarMultError(Exception):
    def __init__(self):
        self.message = "Can not multiply matrix by not int value"

class MatrixRaisingPowerError(Exception):
    def __init__(self):
        self.message = 'Can raise only square matrix to pozitive integer'


class GaussInitError(Exception):
    def __init__(self):
        self.message = "Invalid sizes of matrixes"
        
class GaussE2Error(Exception):
    def __init__(self):
        self.message = "Can not swap one row"

class GaussE3Error(Exception):
    def __init__(self):
        self.message = "Can not multiply one row by zero"

class GaussTestcaseError(Exception):
    def __init__(self):
        self.message = "Incorrect RRANGE, only takes positive"



def mul(leftM, rightM):
    '''
    Умножение слева матрицы leftM на матрицу rightM.\n
    Возвращает матрицу, полученную в результате умножения.
    '''
    if leftM.n != rightM.m:
        raise MatrixMultiplicationError()
    result_values = [[0 for _ in range(rightM.n)] for _ in range(leftM.m)]
    new_n = rightM.n
    new_m = leftM.m
    for i in range(new_m):
        for j in range(new_n):
            result_values[i][j] = sum([leftM.values[i][k] * rightM.values[k][j] for k in range(leftM.n)])
    return Matrix(new_m, new_n, result_values)

def trace(M):
    '''
    Считает и возвращает след матрицы.
    '''
    c = min(M.n, M.m)
    result = Fraction(0, 1)
    for i in range(c):
        result += M.values[i][i]
    return result

def sev_mul(leftM, rightM, *args):
    '''
    Поочередное умножение матриц друг за другом.\n
    Возвращает матрицу, полученную в результате умножений.
    '''
    res = mul(leftM, rightM)
    for arg in args:
        res = mul(res, arg)
    return res


class Matrix:
    '''
    Реализация матрицы размера m на n c заданными значениями.
    '''
    def __init__(self, m: int, n: int, values: list[list[int]]) -> None:
        if type(n) is not int or type(m) is not int:
            raise MatrixInitError()
        if n < 1 or m < 1:
            raise MatrixInitError()
        self.m = m
        self.n = n
        if len(values) != self.m:
            raise MatrixInitError()
        if not all(len(row) == self.n for row in values):
            raise MatrixInitError()
        self.values = values
    
    def __repr__(self) -> str:
        ret_str = f"Size: {self.m}x{self.n}\n"
        for row in self.values:
            print(*row)
        return ret_str
    
    def __add__(self, M):
        if not (self.m == M.m and self.n == M.n):
            raise MatrixAdditionError()
        result_values = deepcopy(self.values)
        for i in range(self.m):
            for j in range(self.n):
                result_values[i][j] += M.values[i][j]
        return Matrix(self.m, self.n, result_values)
    
    def __sub__(self, M):
        if not (self.m == M.m and self.n == M.n):
            raise MatrixAdditionError()
        result_values = deepcopy(self.values)
        for i in range(self.m):
            for j in range(self.n):
                result_values[i][j] -= M.values[i][j]
        return Matrix(self.m, self.n, result_values)

    def __neg__(self, k):
        result_values = deepcopy(self.values)
        for i in range(self.m):
            for j in range(self.n):
                result_values[i][j] *= -1
        return Matrix(self.m, self.n, result_values)
        
    def __mul__(self, k):
        result_values = deepcopy(self.values)
        for i in range(self.m):
            for j in range(self.n):
                result_values[i][j] *= k
        return Matrix(self.m, self.n, result_values)

    def __pow__(self, power: int):
        if self.m != self.n:
            raise MatrixRaisingPowerError()
        if type(power) != int:
            raise MatrixRaisingPowerError()
        power -= 1
        result_M = mul(self, self)
        while power != 1:
            result_M = mul(result_M, self)
            power -= 1
        return result_M

    def transpose(self):
        '''
        Транспонирование матрицы, возвращает транспонированную матрицу без изменения исходной.
        '''
        new_values = [[self.values[row][col] for row in range(self.m)] for col in range(self.n)]
        return Matrix(self.n, self.m, new_values)


class GaussianElimination:
    '''
    Реализация алгоритма прямого и обратного ходов Гаусса.\n
    При инициализации принимает матрицу коэффициентов A и матрицу свободных членов b.
    \n
    Для отображения матриц используется модуль tabulate, а для вычислений в рациональных числах модуль fraction
    '''
    def __init__(self, A: Matrix, b: Matrix):
        if A.m != b.m:
            raise GaussInitError()
        if b.n != 1:
            raise GaussInitError()
        self.m = A.m
        self.n = A.n
        self.coef = A.values
        self.free_c = b.values
        self.aug_values = [deepcopy(A.values[row]) + deepcopy(b.values[row]) for row in range(self.m)]

    def __repr__(self) -> str:
        ret_str = f"Size: {self.m}x{self.n}\n"
        ret_values = deepcopy(self.aug_values)
        for i in range(self.m):
            for j in range(self.n + 1):
                ret_values[i][j] = str(ret_values[i][j])
        print(tabulate(ret_values, headers=[f"x{i}" for i in range(1, self.n + 1)] + ["b"], tablefmt="grid"))
        return ret_str
    
    def e1(self, i: int, j: int, a: float, dbg=0) -> None:
        '''
        Элементарное преобразование первого типа.\n
        Прибавляет к i-ой строке j-ую, умножкенную на скаляр a.
        '''
        if dbg:
            print(f"e1({i + 1}, {j + 1}, {a})")
        for k in range(self.n + 1):
            self.aug_values[i][k] += self.aug_values[j][k] * a
        if dbg:
            for row in self.aug_values:
                print(*row)
            print('-' * 50)
    
    def e2(self, i: int, j: int, dbg=0) -> None:
        '''
        Элементарное преобразование второго типа.\n
        Меняет строки i и j местами.
        '''
        if dbg:
            print(f"e2({i + 1}, {j + 1})")
        if i == j:
            raise GaussE2Error()
        self.aug_values[i], self.aug_values[j] = self.aug_values[j], self.aug_values[i]
        if dbg:
            for row in self.aug_values:
                print(*row)
            print('-' * 50)
    
    def e3(self, i: int, a: float, dbg=0) -> None:
        '''
        Элементарное преобразование третьего типа.\n
        Умножение i-ой строки на ненулевой скаляр a.
        '''
        if dbg:
            print(f"e2({i + 1}, {a})")
        if a == 0:
            raise GaussE3Error()
        self.aug_values[i] = [num * a for num in self.aug_values[i]]
        if dbg:
            for row in self.aug_values:
                print(*row)
            print('-' * 50)
        
    def row_ech(self, row=0, c=0, dbg=0):
        '''
        Алгоритм прямого хода Гаусса для приведения расширенной матрицы к ступенчатому виду.\n
        Ничего не возвращает. Изменяет исходную расширенную матрицу self.aug_values,\n заданной при инициализации объекта.
        '''
        if row >= self.m:
            return
        if c >= self.n:
            return
        if all(self.aug_values[i][j] == 0 for i in range(self.m) for j in range(self.n)):
            return
        while c < self.n and all(self.aug_values[i][c] == 0 for i in range(self.n)):
            c += 1
        if c == self.n:
            return
        if self.aug_values[row][c] == 0:
            k = row + 1
            while k < self.m:
                if self.aug_values[k][c] != 0:
                    self.e2(c, k, dbg)
                    break
                k += 1
        for k in range(row + 1, self.m):
            try:
                self.e1(k, row, -Fraction(self.aug_values[k][c], self.aug_values[row][c]), dbg)
            except ZeroDivisionError:
                pass     
        self.row_ech(row + 1, c + 1, dbg)
        
    def reduced_row_ech(self, dbg=0):
        '''
        Алгоритм обратного хода Гаусса для приведения расширенной матрицы\n к улучшенному ступенчатому виду.
        Ничего не возвращает. \nИзменяет исходную расширенную матрицу self.aug_values, заданной при инициализации объекта.
        '''
        for i in range(self.m):
            for elem in self.aug_values[i]:
                if elem != 0:
                    self.e3(i, Fraction(1, elem), dbg)
                    break
        for i in range(self.m - 1, -1, -1):
            rl = 0
            while self.aug_values[i][rl] != 1 and rl < self.n:
                rl += 1
            if rl >= self.n:
                continue
            for j in range(i - 1, -1, -1):
                self.e1(j, i, -self.aug_values[j][rl], dbg)
        count_zero_rows = self.aug_values.count([0 for _ in range(self.n + 1)])
        c = deepcopy(count_zero_rows)
        while c > 0:
            self.aug_values.remove([0 for _ in range(self.n + 1)])
            c -= 1
        while count_zero_rows > 0:
            self.aug_values.append([0 for _ in range(self.n + 1)])
            count_zero_rows -= 1 
        
    def solve(self, dbg=0, ):
        '''
        Метод для решения СЛУ. \nВ нем вызываются методы row_ech и reduced_row_ech.
        Параметр dbg отвечает за вызов метода с подробным решением,\n в котором отображаются элементарные преобразования.
        '''
        print('-' * 50)
        print('Input matrix')
        print(self)
        self.row_ech(0, 0, dbg)
        print(f"Row echelon form: ")
        print(self)
        self.reduced_row_ech(dbg)
        print('Reduced echelon form: ')
        print(self)
        print('-' * 50)

    @staticmethod
    def generate_testcase(m: int, n: int, RRANGE=100):
        '''
        Генерирует расширенную матрицу размером m на n + 1 со случайными коэффициентами.\n
        Разброс случайных значений котроллируется RRANGE=100.
        Возвращает объект класса GaussianElimination.
        '''
        if RRANGE < 1:
            raise GaussTestcaseError()
        A = [[randrange(-RRANGE, RRANGE) for _ in range(n)] for _ in range(m)]
        b = [[randrange(-RRANGE, RRANGE)] for _ in range(m)]
        return GaussianElimination(Matrix(m, n, A), Matrix(m, 1, b))



