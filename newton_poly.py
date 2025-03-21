def newton_poly(x_nodes, y_nodes):
    n = len(x_nodes)
    assert len(y_nodes) == n
    assert n >= 1
    
    table = []
    table.append([y_nodes[0]])
    
    # 计算差商表
    for i in range(1, n):
        ci = []
        ci.append(y_nodes[i])
        for j in range(1, n):
            val = (ci[j - 1] - table[i - 1][j - 1]) / (x_nodes[i] - x_nodes[i - j])
            ci.append(val)
        table.append(ci)
    
    c = []
    for i in range(n):
        c.append(table[i][i])
    
    def interpolates(x):
        phi = []
        phi.append(1)
        
        for i in range(1, n):
            phi.append(phi[i - 1] * (x - x_nodes[i - 1]))
        
        total = 0.0
        for i in range(n):
            total += c[i] * phi[i]
            
        return total
        
    return interpolates


if __name__ == "__main__":
    xs = [1, 3]
    ys = [3, 4]
    poly = newton_poly(xs, ys)
    print(poly(1))
    print(poly(10))
    print(poly(3))
    