import numpy as np
from numpy import sqrt

print("Hello world")
x = sqrt(2)

print(x)

print(np.cos(1))

ans = 'y'
while ans != 'n':
    outcome = np.random.randint(1,6+1)
    if outcome == 6:
        print("Hooray a 6!")
        break
    else:
        print("Bad luck, a", outcome)
    ans = input("Again? (y/n)")