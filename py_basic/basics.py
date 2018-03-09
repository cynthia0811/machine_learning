import numpy as np 
import matplotlib.pyplot as plt

def sin():
    x = np.linspace(0, 10)
    print(x)
    y = np.sin(x)
    print(y)
    fig, ax = plt.subplots()
    ax.plot(x, y, label="$sin(x)$", color="red", linewidth=1)

    plt.show()


def main():
    items=["aa","bb","cc","dd"]
    for  i,v in enumerate(items):
        print(i,v)
    
    print("\n")

    names = ['Bob', 'Alice', 'Guido']
    # 索引从1开始
    for index, value in enumerate(names, 1):
        print(f'{index}: {value}')


if __name__ == '__main__':
      sin()
