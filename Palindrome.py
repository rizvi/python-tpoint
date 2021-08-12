value = 3534
prevValue = value
remainder = 0
finalValue = 0

while value > 0:
    remainder = value%10
    finalValue = finalValue * 10 + remainder
    value = int(value/10)

if prevValue == finalValue:
    print(finalValue, 'is palindrome')
else:
    print(finalValue, 'is not palindrome')