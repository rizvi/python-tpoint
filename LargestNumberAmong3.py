firstNumber = 555
secondNumber = 888
thirdNumber = 555

if firstNumber > secondNumber:
    if firstNumber > thirdNumber:
        print(firstNumber, 'is the largest number')
    else:
        print(thirdNumber, "is the largest number")
else:
    if secondNumber > thirdNumber:
        print(secondNumber, 'is the largest number')
    else:
        print(thirdNumber, 'is the largest number')