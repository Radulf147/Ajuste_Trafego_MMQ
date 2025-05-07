import csv
import numpy as np

def ler_csv(caminho, x_col, y_col):
    x, y = [], []
    with open(caminho, 'r') as arquivo:
        leitor = csv.DictReader(arquivo)
        for linha in leitor:
            try:
                x.append(float(linha[x_col]))
                y.append(float(linha[y_col]))
            except:
                continue  # ignora linhas com dados inválidos
    return np.array(x), np.array(y)

def ajustar_linear(x, y):
    n = len(x)
    A = np.array([[np.sum(x**2), np.sum(x)],
                  [np.sum(x),     n]])
    B = np.array([np.sum(x*y), np.sum(y)])
    coef = np.linalg.solve(A, B)  # [a, b]
    return coef

def ajustar_cubico(x, y):
    n = len(x)
    X = np.vstack([x**3, x**2, x, np.ones(n)]).T  # matriz do sistema
    XtX = X.T @ X
    Xty = X.T @ y
    coef = np.linalg.solve(XtX, Xty)  # [a, b, c, d]
    return coef

def calcular_R2(y, y_pred, p):
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - ss_res / ss_tot
    r2_ajustado = 1 - (1 - r2)*(len(y) - 1)/(len(y) - p - 1)
    return r2, r2_ajustado

def prever_linear(x, coef):
    return coef[0]*x + coef[1]

def prever_cubico(x, coef):
    return coef[0]*x**3 + coef[1]*x**2 + coef[2]*x + coef[3]

def rodar_analise(caminho_csv):
    # Linear: fluxo = a * densidade + b
    x1, y1 = ler_csv(caminho_csv, 'density', 'flow')
    coef_linear = ajustar_linear(x1, y1)
    y1_pred = prever_linear(x1, coef_linear)
    r2_l, r2_aj_l = calcular_R2(y1, y1_pred, p=1)

    print(f"Ajuste Linear - {caminho_csv}")
    print(f"Coeficientes: a = {coef_linear[0]}, b = {coef_linear[1]}")
    print(f"R² = {r2_l}, R² ajustado = {r2_aj_l}\n")

    # Cúbico: velocidade = ax³ + bx² + cx + d
    x2, y2 = ler_csv(caminho_csv, 'density', 'velocity')
    coef_cubico = ajustar_cubico(x2, y2)
    y2_pred = prever_cubico(x2, coef_cubico)
    r2_c, r2_aj_c = calcular_R2(y2, y2_pred, p=3)

    print(f"Ajuste Cúbico - {caminho_csv}")
    print(f"Coeficientes: a = {coef_cubico[0]}, b = {coef_cubico[1]}, c = {coef_cubico[2]}, d = {coef_cubico[3]}")
    print(f"R² = {r2_c}, R² ajustado = {r2_aj_c}\n")

# Rodar para os dois arquivos
rodar_analise('dataset-foto.csv')
rodar_analise('dataset-ponto.csv')
