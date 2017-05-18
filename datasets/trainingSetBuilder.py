import random

if __name__ == "__main__":
    # For Level 1 - 4
    levels = {
        '1': [
                [6, 10, 96, 100]
            ],
        '2': [
                [1, 5, 89, 95],
            ],
        '3': [
                [1, 10, 82, 86],
                [11, 14, 89, 95],
            ],
        '4': [
                [1, 10, 75, 81],
                [11, 15, 82, 88],
                [16, 20, 89, 95],
            ],
        '5': [
                [1, 10, 68, 74],
                [11, 15, 75, 81],
                [16, 20, 82, 88],
                [21, 25, 89, 95],
            ],
        '6': [
                [1, 10, 61, 67],
                [11, 15, 68, 74],
                [16, 20, 75, 81],
                [21, 25, 82, 88],
                [26, 30, 89, 95],
            ],
        '7': [
                [1, 10, 54, 60],
                [11, 15, 61, 67],
                [16, 20, 68, 74],
                [21, 25, 75, 81],
                [26, 30, 82, 88],
                [31, 35, 89, 95],
            ],
        '8': [
                [1, 10, 48, 53],
                [11, 15, 54, 60],
                [16, 20, 61, 67],
                [21, 25, 68, 74],
                [26, 30, 75, 81],
                [31, 35, 82, 88],
                [36, 40, 89, 95],
            ],
    }
    print(levels['3'])
    with open('levelsData.txt', 'w') as data:
        for i in range(1, 3):
            i = str(i)
            iters = len(levels[i])
            print iters
            for j in range(iters):
                print(levels[i])
                Tl = levels[i][j][0]
                print Tl
                Tr = levels[i][j][1]
                print Tr
                Al = levels[i][j][2]
                print Al
                Ar = levels[i][j][3]
                print Ar
                for k in range(40):
                    arr = map(str, [random.randint(Tl, Tr), random.randint(Al, Ar), i])
                    data.write(",".join(arr)+"\n")
