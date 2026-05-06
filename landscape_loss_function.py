"""
Визуализация ландшафта функции потерь (MSE) для линейной регрессии.
Создает 3D поверхность функции потерь в зависимости от параметров w и b.
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Генерация сетки параметров
w = np.arange(-10, 30, 1)
b = np.arange(-10, 10, 1)

# Генерация данных
np.random.seed(42)  # Для воспроизводимости
x = np.random.rand(100, 1)
y = 2 + 6 * x + (np.random.rand(100, 1) - 0.5)

# Создание сетки для параметров
ww, bb = np.meshgrid(w, b)

# Вычисление функции потерь (MSE) для каждой комбинации параметров
loss = np.zeros_like(ww)
for i in range(ww.shape[0]):
    for j in range(ww.shape[1]):
        y_pred = ww[i, j] * x + bb[i, j]
        loss[i, j] = np.sum((y - y_pred) ** 2) / len(x)


def show_3d(xx, yy, zz, fig, title="MSE", zlim=(0, 20), alpha=0.5):
    """
    Отображает 3D поверхность функции потерь.
    
    Parameters:
    -----------
    xx : array-like
        Сетка значений для оси X (параметр w)
    yy : array-like
        Сетка значений для оси Y (параметр b)
    zz : array-like
        Значения функции потерь (MSE)
    fig : matplotlib.figure.Figure
        Фигура для отображения
    title : str
        Заголовок графика
    zlim : tuple
        Пределы оси Z
    alpha : float
        Прозрачность поверхности
    """
    ax = fig.add_subplot(121, projection="3d")
    surf = ax.plot_surface(xx, yy, zz, cmap=plt.cm.RdYlGn_r, alpha=alpha)
    
    # Контурная проекция на плоскость
    ax.contourf(xx, yy, zz, zdir='z', offset=zlim[0], cmap="RdYlGn_r", alpha=alpha)
    ax.set_zlim(zlim)
    
    ax.set_xlabel("w", fontsize=12)
    ax.set_ylabel("b", fontsize=12)
    ax.set_title(title, fontsize=14)
    fig.colorbar(surf, ax=ax, location="left", shrink=0.5)


# Создание фигуры и отображение
fig = plt.figure(figsize=(15, 5))
show_3d(ww, bb, loss, fig, title="MSE Landscape")

plt.tight_layout()
plt.show()
