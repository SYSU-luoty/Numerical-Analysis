import numpy as np
import matplotlib.pyplot as plt
import random

def lagrange_polynomial(x_nodes, y_nodes):
    """
    构造n个节点的拉格朗日插值多项式
    :param x_nodes: 已知点的x坐标列表，长度需 ≥1 且无重复值
    :param y_nodes: 已知点的y坐标列表，长度需与x_nodes一致
    :return: 插值函数，可计算任意x处的插值结果
    """
    n = len(x_nodes)
    assert len(y_nodes) == n, "x_nodes和y_nodes长度必须相同"
    assert n >= 1, "至少需要1个节点"
    
    # 预计算分母（优化重复计算）
    denominators = []
    for i in range(n):
        denom = 1.0
        for j in range(n):
            if j != i:
                denom *= (x_nodes[i] - x_nodes[j])
        denominators.append(denom)
    
    def interpolate(x):
        total = 0.0
        for i in range(n):
            numerator = 1.0
            for j in range(n):
                if j != i:
                    numerator *= (x - x_nodes[j])
            total += y_nodes[i] * (numerator / denominators[i])
        return total
    
    return interpolate

def generate_polynomial(n, coeff_range=(-5, 5)):
    """
    生成一个n次多项式函数
    :param n: 多项式次数（必须≥0）
    :param coeff_range: 系数生成范围（tuple）
    :return: (多项式函数, 多项式字符串表示)
    """
    assert n >= 0, "多项式次数必须≥0"
    # 生成随机系数，确保最高次项系数非零
    coeff = [random.uniform(*coeff_range) for _ in range(n)]
    coeff.append(random.choice([-1, 1]) * random.uniform(1, coeff_range[1]))  # 最高次项系数
    
    # 构建多项式函数
    def polynomial(x):
        total = 0.0
        for i in range(n+1):
            total += coeff[i] * (x**i)
        return total
    
    # 生成可读的数学表达式
    terms = []
    for i in range(n+1):
        if coeff[i] == 0:
            continue
        term = ""
        if i == 0:
            term = f"{coeff[i]:.2f}"
        else:
            sign = "+" if coeff[i] > 0 else "-"
            value = abs(coeff[i])
            if value == 1 and i > 0:
                value_str = ""
            else:
                value_str = f"{value:.2f}"
            term = f"{sign} {value_str}x^{i}" if i > 1 else f"{sign} {value_str}x"
        terms.append(term)
    
    # 整理表达式
    expr = " ".join(terms[::-1]).replace("x^1", "x").replace("x^0", "").replace("+ -", "- ")
    if expr.startswith("+ "):
        expr = expr[2:]
    expr = expr or "0"  # 处理0多项式情况
    
    return polynomial, expr

def Runge_Phenomenon():
    def polynomial(x):
        return 1 / (1 + 25 * x ** 2)
    return polynomial, "Runge_Phenomenon"

def plot_lagrange(x_nodes, y_nodes, interpolator, func=None, func_name=None, x_range=None, save_path=None):
    """
    绘制拉格朗日插值结果及原函数
    :param x_nodes: 已知节点的x坐标列表
    :param y_nodes: 已知节点的y坐标列表
    :param interpolator: 插值函数
    :param func: 原函数
    :param func_name: 原函数名称
    :param x_range: 自定义绘图范围
    :param save_path: 图片保存路径
    """
    # 确定绘图范围
    if x_range is not None:
        x_min_plot, x_max_plot = x_range
    else:
        x_min = min(x_nodes)
        x_max = max(x_nodes)
        padding = 0.1 * (x_max - x_min) if x_max != x_min else 1.0
        x_min_plot = x_min - padding
        x_max_plot = x_max + padding
    
    x_plot = np.linspace(x_min_plot, x_max_plot, 200)
    y_plot = [interpolator(x) for x in x_plot]
    
    # 创建画布
    plt.figure(figsize=(16, 8), dpi=100)
    
    # 绘制原函数
    if func is not None:
        y_origin = func(x_plot)
        plt.plot(x_plot, y_origin, '--', color='green', label=f'Original Function: {func_name}')
        plt.plot(x_plot, y_origin, '--', color='green')
    
    # 绘制插值曲线
    plt.plot(x_plot, y_plot, color='royalblue', label='Lagrange Interpolation')
    
    # 绘制已知点
    plt.scatter(x_nodes, y_nodes, color='red', zorder=3, 
                label='Interpolation Nodes', s=80, edgecolor='black')
    
    # 装饰图形
    title = f"Lagrange Interpolation of {func_name} ({len(x_nodes)} nodes)"
    # plt.title(title, fontsize=12)
    plt.xlabel('x', fontsize=10)
    plt.ylabel('y', fontsize=10)
    plt.grid(alpha=0.3, linestyle='--')
    plt.legend()
    plt.tight_layout()
    
    # 输出结果
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
    else:
        plt.show()

if __name__ == "__main__":
    # 参数设置
    n_degree = 17      # 多项式次数
    n_nodes = 10          # 插值节点数量
    x_min, x_max = -1, 1  # 节点生成范围
    
    # 生成随机多项式
    # selected_func, func_name = generate_polynomial(n_degree, coeff_range=(-3, 3))
    selected_func, func_name = Runge_Phenomenon()
    
    # 生成插值节点（确保不重复）
    xs = []
    # while len(xs) < n_nodes:
    #     x = np.round(np.random.uniform(x_min, x_max), decimals=2)
    #     if x not in xs:
    #         xs.append(x)
    # xs = np.sort(xs)
    # xs = [-1.0, -0.8, -0.6, -0.4, -0.2, 0.0,  0.2, 0.4, 0.6, 0.8, 1.0]
    # start = -1.0
    start = -1.0
    stop = 1.0  # 原终止值
    step = 0.2
    
    xs = np.arange(start, stop + step, step)
    ys = [selected_func(x) for x in xs]
    
    # 生成插值多项式
    poly = lagrange_polynomial(xs, ys)
    
    # 绘制图像（调整显示范围）
    plot_lagrange(xs, ys, poly,
                  func=selected_func,
                  func_name=func_name,
                  x_range=(x_min, x_max),
                  save_path="poly_interpolation.png")