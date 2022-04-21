import numpy as np

print("low value alpha (1,10)")
for i in range(3):
    print(np.random.beta(1,1))

print("low value beta (10,1)")
for i in range(3):
    print(np.random.beta(10,0.1))


print("high value alpha (100,1)")
for i in range(3):
    print(np.random.beta(100,1))


print("high value beta (1,100)")
for i in range(3):
    print(np.random.beta(1,2))