import numpy as np
import matplotlib.pyplot as plt

# --- Ввод данных ---
choice = input("Вы хотите ввести данные из файла? (y/n): ").strip().lower()

if choice == 'y':
    filename = input("Введите имя файла: ").strip()
    with open(filename, 'r') as f:
        x = np.array(list(map(float, f.read().strip().split())))
else:
    use_default = input("Использовать значения по умолчанию? (y/n): ").strip().lower()
    if use_default == 'y':
        x = np.linspace(0, 4, 11)
    else:
        x_input = input("Введите значения x через пробел: ")
        x = np.array(list(map(float, x_input.strip().split())))

# Проверка ОДЗ для логарифмической и степенной
x_valid_log = x[x > 0]
x_valid_pow = x[x != 0]

# Целевая функция
y = 18 * x / (x**4 + 10)

results = {}

# --- Линейная аппроксимация ---
SX = np.sum(x)
SXX = np.sum(x**2)
SY = np.sum(y)
SXY = np.sum(x * y)
n = len(x)

A = np.array([[SXX, SX], [SX, n]])
B = np.array([SXY, SY])
[a, b] = np.linalg.solve(A, B)
phi = a * x + b
s = np.sum((phi - y) ** 2)
sigma = np.sqrt(s / n)
y_mean = np.mean(y)
r2 = 1 - s / np.sum((y - y_mean) ** 2)
r = np.corrcoef(x, y)[0, 1]

results["Линейная"] = {
    "params": [a, b],
    "phi": phi,
    "s": s,
    "sigma": sigma,
    "r2": r2,
    "r": r,
    "x": x,
    "y": y,
    "eps": phi - y
}

# --- Квадратичная аппроксимация (2-я степень) ---
Sx = np.sum(x)
Sx2 = np.sum(x**2)
Sx3 = np.sum(x**3)
Sx4 = np.sum(x**4)
Sy = np.sum(y)
Sxy = np.sum(x * y)
Sx2y = np.sum((x**2) * y)

A = np.array([
    [n, Sx, Sx2],
    [Sx, Sx2, Sx3],
    [Sx2, Sx3, Sx4]
])
B = np.array([Sy, Sxy, Sx2y])
a0, a1, a2 = np.linalg.solve(A, B)
phi = a0 + a1 * x + a2 * x**2
s = np.sum((phi - y) ** 2)
sigma = np.sqrt(s / n)
r2 = 1 - s / np.sum((y - y_mean) ** 2)

results["Квадратичная"] = {
    "params": [a0, a1, a2],
    "phi": phi,
    "s": s,
    "sigma": sigma,
    "r2": r2,
    "r": None,
    "x": x,
    "y": y,
    "eps": phi - y
}

# --- Полиномиальная 3-й степени аппроксимация ---
Sx = np.sum(x)
Sx2 = np.sum(x**2)
Sx3 = np.sum(x**3)
Sx4 = np.sum(x**4)
Sx5 = np.sum(x**5)
Sx6 = np.sum(x**6)
Sy = np.sum(y)
Sxy = np.sum(x * y)
Sx2y = np.sum(x**2 * y)
Sx3y = np.sum(x**3 * y)

A = np.array([
    [n, Sx, Sx2, Sx3],
    [Sx, Sx2, Sx3, Sx4],
    [Sx2, Sx3, Sx4, Sx5],
    [Sx3, Sx4, Sx5, Sx6]
])
B = np.array([Sy, Sxy, Sx2y, Sx3y])
a0, a1, a2, a3 = np.linalg.solve(A, B)
phi = a0 + a1 * x + a2 * x**2 + a3 * x**3
s = np.sum((phi - y) ** 2)
sigma = np.sqrt(s / n)
r2 = 1 - s / np.sum((y - y_mean) ** 2)

results["Полином 3-й степени"] = {
    "params": [a0, a1, a2, a3],
    "phi": phi,
    "s": s,
    "sigma": sigma,
    "r2": r2,
    "r": None,
    "x": x,
    "y": y,
    "eps": phi - y
}

# --- Степенная аппроксимация (через линейную регрессию в логарифмах) ---
X = np.log(x_valid_pow)
Y = np.log(18 * x_valid_pow / (x_valid_pow**4 + 10))
SX = np.sum(X)
SXX = np.sum(X**2)
SY = np.sum(Y)
SXY = np.sum(X * Y)
n_pow = len(X)

A = np.array([[SXX, SX], [SX, n_pow]])
B = np.array([SXY, SY])
B_, A_ = np.linalg.solve(A, B)
a = np.exp(A_)
b = B_
phi = a * x_valid_pow**b
s = np.sum((phi - 18 * x_valid_pow / (x_valid_pow**4 + 10))**2)
sigma = np.sqrt(s / n_pow)
r2 = 1 - s / np.sum((18 * x_valid_pow / (x_valid_pow**4 + 10) - np.mean(18 * x_valid_pow / (x_valid_pow**4 + 10)))**2)

results["Степенная"] = {
    "params": [a, b],
    "phi": phi,
    "s": s,
    "sigma": sigma,
    "r2": r2,
    "r": None,
    "x": x_valid_pow,
    "y": 18 * x_valid_pow / (x_valid_pow**4 + 10),
    "eps": phi - 18 * x_valid_pow / (x_valid_pow**4 + 10)
}

# --- Экспоненциальная аппроксимация ---
epsilon = 1e-10
X = x
Y = np.log(18 * x / (x**4 + 10) + epsilon)
SX = np.sum(X)
SXX = np.sum(X**2)
SY = np.sum(Y)
SXY = np.sum(X * Y)

A = np.array([[SXX, SX], [SX, n]])
B = np.array([SXY, SY])
B_, A_ = np.linalg.solve(A, B)
a = np.exp(A_)
b = B_
phi = a * np.exp(b * x)
s = np.sum((phi - y)**2)
sigma = np.sqrt(s / n)
r2 = 1 - s / np.sum((y - y_mean) ** 2)

results["Экспоненциальная"] = {
    "params": [a, b],
    "phi": phi,
    "s": s,
    "sigma": sigma,
    "r2": r2,
    "r": None,
    "x": x,
    "y": y,
    "eps": phi - y
}

# --- Логарифмическая аппроксимация ---
X = np.log(x_valid_log + epsilon)
Y = 18 * x_valid_log / (x_valid_log**4 + 10)
SX = np.sum(X)
SXX = np.sum(X**2)
SY = np.sum(Y)
SXY = np.sum(X * Y)
n_log = len(X)
X = np.log(x_valid_log)
Y = 18 * x_valid_log / (x_valid_log**4 + 10)
SX = np.sum(X)
SXX = np.sum(X**2)
SY = np.sum(Y)
SXY = np.sum(X * Y)
n_log = len(X)

A = np.array([[SXX, SX], [SX, n_log]])
B = np.array([SXY, SY])
a, b = np.linalg.solve(A, B)
phi = a * np.log(x_valid_log) + b
s = np.sum((phi - Y)**2)
sigma = np.sqrt(s / n_log)
r2 = 1 - s / np.sum((Y - np.mean(Y)) ** 2)

results["Логарифмическая"] = {
    "params": [a, b],
    "phi": phi,
    "s": s,
    "sigma": sigma,
    "r2": r2,
    "r": None,
    "x": x_valid_log,
    "y": Y,
    "eps": phi - Y
}

rounded = lambda arr: [round(float(val), 3) for val in arr]
# --- Вывод ---
for name, result in results.items():

    print(f"\n{name} аппроксимация:")
    rounded_params = rounded(result['params'])
    print(f"  Коэффициенты: {rounded_params}")
    print(f"  S: {result['s']:.3f}")
    print(f"  σ: {result['sigma']:.3f}")
    print(f"  R²: {result['r2']:.3f}")
    if result['r'] is not None:
        print(f"  r: {result['r']:.3f}")
    print(f"  xi: {rounded(result['x'])}")
    print(f"  yi: {rounded(result['y'])}")
    print(f"  φ(xi): {rounded(result['phi'])}")
    print(f"  εi: {rounded(result['eps'])}")

# --- График ---
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='black', label='Исходные данные')
for name, result in results.items():
    plt.plot(result['x'], result['phi'], label=f"{name} (σ ≈ {result['sigma']:.3f})")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Аппроксимация")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
