import numpy as np
from scipy.optimize import curve_fit

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
    print(f'welcom to use {name} !')
    print(f"测试git和pycharm如何使用！")

# 创建函数f(x) = ax + b
def func(x, a, b):
    return a * x + b


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

    # 创建干净数据
    x = np.linspace(0, 10, 100)
    y = func(x, 1, 2)
    # 添加噪声
    yn = y + 0.9 * np.random.normal(size=len(x))

    # 拟合噪声数据
    popt, pcov = curve_fit(func, x, yn)
    # 输出最优参数
    print(popt)