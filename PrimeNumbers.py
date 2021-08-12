import math

# To take input from the user
inputVal = int(input("Enter a number: "))
# prime numbers are greater than 1
if inputVal > 1:
    for num in range(3, inputVal, 2):
        # check for factors
        for i in range(2, math.floor(math.sqrt(num))):
            if (num % i) == 0:
                print(num, "is not a prime number")
                print(i, "times", num // i, "is", num)
                break
        else:
            print(num, "is a prime number")
