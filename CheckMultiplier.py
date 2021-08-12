from typing import Final


def multiply(x, y=9):
    print('value of x', x)
    print('value of y', y)
    return x * y


print(multiply(2, 3))
print(multiply(22))
print(multiply(y=2, x=5))

a: Final = 1
a = 2

print('a is:', a)
