import csv
from random import randint, uniform


TRAIN_N = 200

TEST_N = 100


with open('train.csv', 'w') as csvfile:

    writer = csv.writer(csvfile)
    writer.writerow([TRAIN_N,2,0,1])
    for i in range(TRAIN_N):
        y = randint(0,1)
        x1 = uniform(0,1)
        x2 = 0
        if y is 0 and x1 < 0.5:
            x2 = uniform(0, 0.6)
        elif y is 0 and x1 > 0.5:
            x2 = uniform(0.4, 1)
        elif y is 1 and x1 < 0.5:
            x2 = uniform(0.4, 1)
        else:
            x2 = uniform(0, 0.6)
        writer.writerow([x1,x2,y])


with open('test.csv', 'w') as csvfile:

    writer = csv.writer(csvfile)
    writer.writerow([TEST_N,2,0,1])
    for i in range(TEST_N):
        y = randint(0,1)
        x1 = uniform(0,1)
        x2 = 0
        if y is 0 and x1 < 0.5:
            x2 = uniform(0, 0.6)
        elif y is 0 and x1 > 0.5:
            x2 = uniform(0.4, 1)
        elif y is 1 and x1 < 0.5:
            x2 = uniform(0.4, 1)
        else:
            x2 = uniform(0, 0.6)
        writer.writerow([x1,x2,y])


