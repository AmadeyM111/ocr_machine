"""
Визуализация функции y = x² - x + 3 с отмеченной точкой минимума.

Этот скрипт создает график квадратичной функции и отмечает точку минимума,
где производная равна нулю (x = 0.5, y = 2.75).
"""
import numpy as np
import matplotlib.pyplot as plt

# Параметры визуализации
X_MIN = -2
X_MAX = 3
NUM_POINTS = 100
TICK_STEP = 0.5

# Параметры функции: y = x² - x + 3
# Коэффициенты: a=1, b=-1, c=3
# Точка минимума: x = -b/(2a) = 1/2 = 0.5
MIN_X = 0.5
MIN_Y = MIN_X**2 - MIN_X + 3  # Вычисляем y для точки минимума

# Генерация данных
x = np.linspace(X_MIN, X_MAX, NUM_POINTS)
y = x**2 - x + 3

# Создание графика
fig, ax = plt.subplots(figsize=(10, 6))
fig.suptitle("$y = x^2 - x + 3$", fontsize=14, fontweight='bold')

# Построение кривой
ax.plot(x, y, linewidth=2, label="$y = x^2 - x + 3$")

# Настройка осей
ax.set(xticks=np.arange(X_MIN, X_MAX + TICK_STEP, step=TICK_STEP))
ax.set_xlabel("$x$", fontsize=12)
ax.set_ylabel("$y$", fontsize=12)
ax.grid(True, alpha=0.3, linestyle='--')

# Отметка точки минимума
ax.plot(MIN_X, MIN_Y, marker="o", markersize=8, color='red', 
        label=f"Минимум: ({MIN_X}, {MIN_Y})", zorder=5)
ax.axvline(MIN_X, color='red', linestyle=':', alpha=0.5, linewidth=1)
ax.axhline(MIN_Y, color='red', linestyle=':', alpha=0.5, linewidth=1)

# Добавление легенды
ax.legend(loc='best', fontsize=10)

# Улучшение внешнего вида
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Сохранение графика (опционально)
# Раскомментируйте следующую строку для сохранения:
# plt.savefig('differentiation_visualization.png', dpi=300, bbox_inches='tight')

# Отображение графика
plt.tight_layout()
plt.show()