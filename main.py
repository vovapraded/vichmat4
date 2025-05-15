import numpy as np
import matplotlib.pyplot as plt

def get_function_string(name, params):
    if name == "Линейная":
        return f"φ(x) = {params[0]:.3f} * x + {params[1]:.3f}"
    elif name == "Квадратичная":
        return f"φ(x) = {params[0]:.3f} + {params[1]:.3f} * x + {params[2]:.3f} * x²"
    elif name == "Полином 3-й степени":
        return f"φ(x) = {params[0]:.3f} + {params[1]:.3f} * x + {params[2]:.3f} * x² + {params[3]:.3f} * x³"
    elif name == "Степенная":
        return f"φ(x) = {params[0]:.3f} * x^{params[1]:.3f}"
    elif name == "Экспоненциальная":
        return f"φ(x) = {params[0]:.3f} * e^({params[1]:.3f} * x)"
    elif name == "Логарифмическая":
        return f"φ(x) = {params[0]:.3f} * ln(x) + {params[1]:.3f}"
    return "Неизвестная модель"

def sorted_result(x, y, phi):
    idx = np.argsort(x)
    x_sorted = x[idx]
    y_sorted = y[idx]
    phi_sorted = phi[idx]
    eps_sorted = phi_sorted - y_sorted
    return x_sorted, y_sorted, phi_sorted, eps_sorted

# --- Ввод данных ---
choice = input("Вы хотите ввести данные из файла? (y/n): ").strip().lower()

if choice == 'y':
    filename = input("Введите имя файла: ").strip()
    with open(filename, 'r') as f:
        lines = f.readlines()
        if len(lines) < 2:
            raise ValueError("Файл должен содержать как минимум две строки: x и y.")
        x = np.array(list(map(float, lines[0].strip().split())))
        y = np.array(list(map(float, lines[1].strip().split())))
else:
    use_default = input("Использовать значения по умолчанию? (y/n): ").strip().lower()
    if use_default == 'y':
        x = np.linspace(0, 4, 11)
        y = 18 * x / (x**4 + 10)
    else:
        x_input = input("Введите значения x через пробел: ")
        x = np.array(list(map(float, x_input.strip().split())))
        y_input = input("Введите значения y через пробел (той же длины): ")
        y = np.array(list(map(float, y_input.strip().split())))

if len(x) != len(y):
    raise ValueError("Количество значений x и y должно совпадать.")

results = {}
epsilon = 1e-10
n = len(x)
y_mean = np.mean(y)

# --- Подмножества ---
x_valid_log = x[x > 0]
y_valid_log = y[x > 0]

mask_pow = (x != 0) & (y > 0)
x_valid_pow = x[mask_pow]
y_valid_pow = y[mask_pow]

mask_exp = y > 0
x_valid_exp = x[mask_exp]
y_valid_exp = y[mask_exp]

# --- Линейная ---
SX = np.sum(x)
SXX = np.sum(x**2)
SY = np.sum(y)
SXY = np.sum(x * y)

A = np.array([[SXX, SX], [SX, n]])
B = np.array([SXY, SY])
a, b = np.linalg.solve(A, B)
phi = a * x + b
s = np.sum((phi - y) ** 2)
sigma = np.sqrt(s / n)
r2 = 1 - s / np.sum((y - y_mean) ** 2)
r = np.corrcoef(x, y)[0, 1]
x_sorted, y_sorted, phi_sorted, eps_sorted = sorted_result(x, y, phi)

results["Линейная"] = {
    "params": [a, b], "phi": phi_sorted, "s": s, "sigma": sigma,
    "r2": r2, "r": r, "x": x_sorted, "y": y_sorted, "eps": eps_sorted
}

# --- Квадратичная ---
Sx = SX
Sx2 = SXX
Sx3 = np.sum(x**3)
Sx4 = np.sum(x**4)
Sxy = SXY
Sx2y = np.sum(x**2 * y)

A = np.array([[n, Sx, Sx2], [Sx, Sx2, Sx3], [Sx2, Sx3, Sx4]])
B = np.array([SY, Sxy, Sx2y])
a0, a1, a2 = np.linalg.solve(A, B)
phi = a0 + a1 * x + a2 * x**2
s = np.sum((phi - y)**2)
sigma = np.sqrt(s / n)
r2 = 1 - s / np.sum((y - y_mean)**2)
x_sorted, y_sorted, phi_sorted, eps_sorted = sorted_result(x, y, phi)

results["Квадратичная"] = {
    "params": [a0, a1, a2], "phi": phi_sorted, "s": s, "sigma": sigma,
    "r2": r2, "r": None, "x": x_sorted, "y": y_sorted, "eps": eps_sorted
}

# --- Полином 3-й степени ---
Sx5 = np.sum(x**5)
Sx6 = np.sum(x**6)
Sx3y = np.sum(x**3 * y)

A = np.array([
    [n, Sx, Sx2, Sx3],
    [Sx, Sx2, Sx3, Sx4],
    [Sx2, Sx3, Sx4, Sx5],
    [Sx3, Sx4, Sx5, Sx6]
])
B = np.array([SY, Sxy, Sx2y, Sx3y])
a0, a1, a2, a3 = np.linalg.solve(A, B)
phi = a0 + a1 * x + a2 * x**2 + a3 * x**3
s = np.sum((phi - y)**2)
sigma = np.sqrt(s / n)
r2 = 1 - s / np.sum((y - y_mean)**2)
x_sorted, y_sorted, phi_sorted, eps_sorted = sorted_result(x, y, phi)

results["Полином 3-й степени"] = {
    "params": [a0, a1, a2, a3], "phi": phi_sorted, "s": s, "sigma": sigma,
    "r2": r2, "r": None, "x": x_sorted, "y": y_sorted, "eps": eps_sorted
}

# --- Степенная ---
if len(x_valid_pow) > 0:
    X = np.log(x_valid_pow)
    Y = np.log(y_valid_pow)
    SX = np.sum(X)
    SXX = np.sum(X**2)
    SY = np.sum(Y)
    SXY = np.sum(X * Y)
    n_pow = len(X)

    A = np.array([[SXX, SX], [SX, n_pow]])
    B = np.array([SXY, SY])
    b, loga = np.linalg.solve(A, B)
    a = np.exp(loga)
    phi = a * x_valid_pow ** b
    s = np.sum((phi - y_valid_pow)**2)
    sigma = np.sqrt(s / n_pow)
    r2 = 1 - s / np.sum((y_valid_pow - np.mean(y_valid_pow))**2)
    x_sorted, y_sorted, phi_sorted, eps_sorted = sorted_result(x_valid_pow, y_valid_pow, phi)

    results["Степенная"] = {
        "params": [a, b], "phi": phi_sorted, "s": s, "sigma": sigma,
        "r2": r2, "r": None, "x": x_sorted, "y": y_sorted, "eps": eps_sorted
    }

# --- Экспоненциальная ---
if len(y_valid_exp) > 0:
    X = x_valid_exp
    Y = np.log(y_valid_exp + (x_valid_exp == 0) * epsilon)
    SX = np.sum(X)
    SXX = np.sum(X**2)
    SY = np.sum(Y)
    SXY = np.sum(X * Y)
    n_exp = len(X)

    A = np.array([[SXX, SX], [SX, n_exp]])
    B = np.array([SXY, SY])
    b, loga = np.linalg.solve(A, B)
    a = np.exp(loga)
    phi = a * np.exp(b * x_valid_exp)
    s = np.sum((phi - y_valid_exp)**2)
    sigma = np.sqrt(s / n_exp)
    r2 = 1 - s / np.sum((y_valid_exp - np.mean(y_valid_exp))**2)
    x_sorted, y_sorted, phi_sorted, eps_sorted = sorted_result(x_valid_exp, y_valid_exp, phi)

    results["Экспоненциальная"] = {
        "params": [a, b], "phi": phi_sorted, "s": s, "sigma": sigma,
        "r2": r2, "r": None, "x": x_sorted, "y": y_sorted, "eps": eps_sorted
    }

# --- Логарифмическая ---
if len(x_valid_log) > 0:
    X = np.log(x_valid_log + (x_valid_log == 0) * epsilon)
    Y = y_valid_log
    SX = np.sum(X)
    SXX = np.sum(X**2)
    SY = np.sum(Y)
    SXY = np.sum(X * Y)
    n_log = len(X)

    A = np.array([[SXX, SX], [SX, n_log]])
    B = np.array([SXY, SY])
    a, b = np.linalg.solve(A, B)
    phi = a * X + b
    s = np.sum((phi - Y)**2)
    sigma = np.sqrt(s / n_log)
    r2 = 1 - s / np.sum((Y - np.mean(Y))**2)
    x_sorted, y_sorted, phi_sorted, eps_sorted = sorted_result(x_valid_log, Y, phi)

    results["Логарифмическая"] = {
        "params": [a, b], "phi": phi_sorted, "s": s, "sigma": sigma,
        "r2": r2, "r": None, "x": x_sorted, "y": y_sorted, "eps": eps_sorted
    }

# --- Вывод ---
rounded = lambda arr: [round(float(val), 3) for val in arr]
for name, result in results.items():
    print(f"\n{name} аппроксимация:")
    print(f"  Функция: {get_function_string(name, result['params'])}")
    print(f"  Коэффициенты: {rounded(result['params'])}")
    print(f"  S: {result['s']:.3f}")
    print(f"  σ: {result['sigma']:.3f}")
    print(f"  R²: {result['r2']:.3f}")
    r2 = result['r2']
    if r2 >= 0.95:
        quality = "высокая точность аппроксимации (модель хорошо описывает явление)"
    elif 0.75 <= r2 < 0.95:
        quality = "удовлетворительная аппроксимация (модель адекватно описывает явление)"
    elif 0.5 <= r2 < 0.75:
        quality = "слабая аппроксимация (модель слабо описывает явление)"
    else:
        quality = "недостаточная точность (модель требует изменения)"
    print(f"  Оценка по R²: {quality}")
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

# --- Лучшая аппроксимация по σ ---
min_sigma = min(result["sigma"] for result in results.values())
epsilon_sigma = 1e-6
best_fits = {name: result for name, result in results.items()
             if abs(result["sigma"] - min_sigma) <= epsilon_sigma}

print("\nЛучшие аппроксимации по критерию минимального СКО:")
for name, result in best_fits.items():
    print(f"\n  Тип: {name}")
    print(f"  Функция: {get_function_string(name, result['params'])}")
    print(f"  Коэффициенты: {rounded(result['params'])}")
    print(f"  σ: {result['sigma']:.3f}")
    print(f"  R²: {result['r2']:.3f}")
