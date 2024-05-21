birth_year = input('Birth year: ')
print(type(birth_year))
# age = 2021 - birth_year ==> #TypeError: unsupported operand type(s) for -: 'int' and 'str'
age = 2021 - int(birth_year)
print(age)
print(type(age))

