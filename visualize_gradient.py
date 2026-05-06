import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def show_3d(xx, yy, zz, fig, x, y, xxsmall, yysmall, gradx, grady, title="sin(xy)", zlim=(-2, 2), alpha=0.7):
    """
    Визуализация 3D поверхности функции с градиентом в виде векторов
    
    Parameters:
    -----------
    xx, yy : array-like
        Сетки координат для построения поверхности
    zz : array-like
        2D массив значений функции (высота поверхности)
    fig : matplotlib.figure.Figure
        Фигура для отрисовки
    x, y : array-like
        Границы для осей (min и max значения)
    xxsmall, yysmall : array-like
        Координаты для векторов градиента
    gradx, grady : array-like
        Компоненты градиента (dx, dy)
    title : str
        Заголовок графика
    zlim : tuple
        Пределы значений z для оси
    alpha : float
        Прозрачность поверхности
    """
    ax = fig.add_subplot(111, projection='3d')
    
    # Построение 3D поверхности
    surf = ax.plot_surface(
        xx, yy, zz,
        cmap="RdYlGn_r",
        alpha=alpha,
        linewidth=0,
        antialiased=True
    )
    
    # Вычисление z-координат для векторов градиента на поверхности
    # Находим индексы в исходной сетке для уменьшенной сетки
    step_x = len(zz) // len(xxsmall) if len(zz) > len(xxsmall) else 1
    step_y = len(zz[0]) // len(xxsmall[0]) if len(zz[0]) > len(xxsmall[0]) else 1
    zzsmall = zz[::step_x, ::step_y]
    
    # Обрезаем до нужного размера, если необходимо
    min_shape = (min(zzsmall.shape[0], xxsmall.shape[0]), 
                 min(zzsmall.shape[1], xxsmall.shape[1]))
    zzsmall = zzsmall[:min_shape[0], :min_shape[1]]
    xxsmall_trimmed = xxsmall[:min_shape[0], :min_shape[1]]
    yysmall_trimmed = yysmall[:min_shape[0], :min_shape[1]]
    gradx_trimmed = gradx[:min_shape[0], :min_shape[1]]
    grady_trimmed = grady[:min_shape[0], :min_shape[1]]
    
    # Построение векторов градиента на поверхности
    ax.quiver(
        xxsmall_trimmed, yysmall_trimmed, zzsmall,
        gradx_trimmed, grady_trimmed, np.zeros_like(gradx_trimmed),
        color='blue',
        arrow_length_ratio=0.3,
        length=0.1
    )
    
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(title)
    ax.set_xlim(np.min(x), np.max(x))
    ax.set_ylim(np.min(y), np.max(y))
    ax.set_zlim(zlim)
    
    # Добавление цветовой шкалы
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    return ax

# Пример использования:
fig = plt.figure(figsize=(15, 5))
x = np.linspace(-2, 2, 100)
y = np.linspace(-2, 2, 100)
xx, yy = np.meshgrid(x, y)
zz = np.sin(xx * yy)

# Вычисление градиента
gradx = np.cos(xx * yy) * yy
grady = np.cos(xx * yy) * xx

# Уменьшенная сетка для векторов
step = 5
xxsmall = xx[::step, ::step]
yysmall = yy[::step, ::step]
gradx_small = gradx[::step, ::step]
grady_small = grady[::step, ::step]

show_3d(xx, yy, zz, fig, x, y, xxsmall, yysmall, gradx_small, grady_small, title="sin(xy)")
plt.show()