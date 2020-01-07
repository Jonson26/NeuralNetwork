def factorial(x):#some integer
    i = 0
    p = 1
    while i < x:
        i += 1
        p = p*i
    return p

def exponent(x, i):#power, depth (int)
    j = 0
    s = 0
    while j < i:
        s += (x**j)/factorial(j)
        j += 1
    return s

e = exponent(1, 100)

def sigmoid(x):
    return 1/(1 + e**x)
