"""
Модуль для тестирования стационарности временных рядов.

Поддерживает несколько тестов:
- ADF (Augmented Dickey-Fuller)
- KPSS (Kwiatkowski-Phillips-Schmidt-Shin)
- Phillips-Perron

Автор: Улучшенная версия
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, Dict, Tuple
from statsmodels.tsa.stattools import adfuller, kpss
import warnings

# Попытка импорта Phillips-Perron (может быть недоступен в старых версиях statsmodels)
try:
    from statsmodels.tsa.stattools import PhillipsPerron as pp_test
    PP_AVAILABLE = True
except ImportError:
    try:
        # Альтернативный способ импорта для некоторых версий
        from statsmodels.stats.diagnostic import unitroot_adf
        PP_AVAILABLE = False
    except ImportError:
        PP_AVAILABLE = False


class StationarityTest:
    """
    Класс для проведения тестов стационарности временных рядов.
    """
    
    def __init__(self, data: Union[np.ndarray, pd.Series, list], 
                 alpha: float = 0.05,
                 verbose: bool = True):
        """
        Инициализация теста стационарности.
        
        Args:
            data: Временной ряд для тестирования
            alpha: Уровень значимости (по умолчанию 0.05)
            verbose: Выводить ли подробную информацию
        """
        self.data = self._prepare_data(data)
        self.alpha = alpha
        self.verbose = verbose
        self.results = {}
        
    def _prepare_data(self, data: Union[np.ndarray, pd.Series, list]) -> np.ndarray:
        """
        Подготовка данных для тестирования.
        
        Args:
            data: Входные данные
            
        Returns:
            np.ndarray: Подготовленный массив
        """
        if isinstance(data, pd.Series):
            data = data.values
        elif isinstance(data, list):
            data = np.array(data)
        elif not isinstance(data, np.ndarray):
            raise TypeError(f"Неподдерживаемый тип данных: {type(data)}")
        
        # Удаляем NaN значения
        data = data[~np.isnan(data)]
        
        if len(data) < 10:
            raise ValueError("Недостаточно данных для тестирования (минимум 10 наблюдений)")
        
        return data
    
    def adf_test(self, maxlag: Optional[int] = None, 
                 regression: str = 'c',
                 autolag: str = 'AIC') -> Dict:
        """
        Тест Augmented Dickey-Fuller (ADF).
        
        H0: Временной ряд имеет единичный корень (нестационарен)
        H1: Временной ряд стационарен
        
        Args:
            maxlag: Максимальное количество лагов (None = автоматически)
            regression: Тип регрессии ('c' - константа, 'ct' - константа+тренд, 'ctt' - константа+линейный+квадратичный тренд, 'nc' - без константы)
            autolag: Метод автоматического выбора лагов ('AIC', 'BIC', 't-stat', None)
            
        Returns:
            dict: Результаты теста
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = adfuller(self.data, maxlag=maxlag, regression=regression, autolag=autolag)
        
        # adfuller возвращает:
        # result[0] - статистика ADF
        # result[1] - p-value
        # result[2] - количество лагов
        # result[3] - количество наблюдений
        # result[4] - критические значения
        # result[5] - icbest (если autolag используется)
        
        adf_statistic = result[0]
        p_value = result[1]
        used_lag = result[2]
        n_obs = result[3]
        critical_values = result[4]
        
        is_stationary = p_value <= self.alpha
        
        test_result = {
            'test_name': 'ADF (Augmented Dickey-Fuller)',
            'statistic': adf_statistic,
            'p_value': p_value,
            'used_lag': used_lag,
            'n_obs': n_obs,
            'critical_values': critical_values,
            'is_stationary': is_stationary,
            'alpha': self.alpha,
            'interpretation': self._interpret_adf(adf_statistic, critical_values, p_value)
        }
        
        self.results['adf'] = test_result
        
        if self.verbose:
            self._print_adf_results(test_result)
        
        return test_result
    
    def _interpret_adf(self, statistic: float, critical_values: Dict, p_value: float) -> str:
        """
        Интерпретация результатов ADF теста.
        
        Args:
            statistic: ADF статистика
            critical_values: Критические значения
            p_value: p-value
            
        Returns:
            str: Интерпретация
        """
        # Сравниваем статистику с критическими значениями
        if statistic < critical_values['1%']:
            strength = "сильно"
        elif statistic < critical_values['5%']:
            strength = "умеренно"
        elif statistic < critical_values['10%']:
            strength = "слабо"
        else:
            strength = "не"
        
        if p_value <= self.alpha:
            return f"Ряд стационарен ({strength} стационарен, p-value={p_value:.4f} <= {self.alpha})"
        else:
            return f"Ряд нестационарен (p-value={p_value:.4f} > {self.alpha})"
    
    def kpss_test(self, regression: str = 'c', 
                  nlags: Optional[str] = 'auto') -> Dict:
        """
        Тест KPSS (Kwiatkowski-Phillips-Schmidt-Shin).
        
        H0: Временной ряд стационарен
        H1: Временной ряд нестационарен
        
        ВАЖНО: KPSS имеет противоположные гипотезы по сравнению с ADF!
        
        Args:
            regression: Тип регрессии ('c' - константа, 'ct' - константа+тренд)
            nlags: Количество лагов ('auto' или число)
            
        Returns:
            dict: Результаты теста
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = kpss(self.data, regression=regression, nlags=nlags)
        
        # kpss возвращает:
        # result[0] - статистика KPSS
        # result[1] - p-value
        # result[2] - количество лагов
        # result[3] - критические значения
        
        kpss_statistic = result[0]
        p_value = result[1]
        used_lag = result[2]
        critical_values = result[3]
        
        # Для KPSS: если p-value < alpha, отвергаем H0 (ряд нестационарен)
        is_stationary = p_value >= self.alpha
        
        test_result = {
            'test_name': 'KPSS (Kwiatkowski-Phillips-Schmidt-Shin)',
            'statistic': kpss_statistic,
            'p_value': p_value,
            'used_lag': used_lag,
            'critical_values': critical_values,
            'is_stationary': is_stationary,
            'alpha': self.alpha,
            'interpretation': self._interpret_kpss(kpss_statistic, critical_values, p_value)
        }
        
        self.results['kpss'] = test_result
        
        if self.verbose:
            self._print_kpss_results(test_result)
        
        return test_result
    
    def _interpret_kpss(self, statistic: float, critical_values: Dict, p_value: float) -> str:
        """
        Интерпретация результатов KPSS теста.
        
        Args:
            statistic: KPSS статистика
            critical_values: Критические значения
            p_value: p-value
            
        Returns:
            str: Интерпретация
        """
        # Для KPSS сравниваем статистику с критическими значениями
        if statistic > critical_values['1%']:
            strength = "сильно"
        elif statistic > critical_values['5%']:
            strength = "умеренно"
        elif statistic > critical_values['10%']:
            strength = "слабо"
        else:
            strength = "не"
        
        if p_value >= self.alpha:
            return f"Ряд стационарен (p-value={p_value:.4f} >= {self.alpha})"
        else:
            return f"Ряд нестационарен ({strength} нестационарен, p-value={p_value:.4f} < {self.alpha})"
    
    def pp_test(self, lags: Optional[int] = None,
                regression: str = 'c') -> Optional[Dict]:
        """
        Тест Phillips-Perron.
        
        H0: Временной ряд имеет единичный корень (нестационарен)
        H1: Временной ряд стационарен
        
        Args:
            lags: Количество лагов (None = автоматически)
            regression: Тип регрессии ('c', 'ct', 'ctt', 'nc')
            
        Returns:
            dict: Результаты теста или None если тест недоступен
        """
        if not PP_AVAILABLE:
            if self.verbose:
                print("⚠️  Тест Phillips-Perron недоступен в вашей версии statsmodels.")
                print("   Установите statsmodels >= 0.13.0 для использования этого теста.")
            return None
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = pp_test(self.data, lags=lags, trend=regression)
            
            pp_statistic = result.stat
            p_value = result.pvalue
            critical_values = result.critical_values
            
            is_stationary = p_value <= self.alpha
            
            test_result = {
                'test_name': 'Phillips-Perron',
                'statistic': pp_statistic,
                'p_value': p_value,
                'critical_values': critical_values,
                'is_stationary': is_stationary,
                'alpha': self.alpha,
                'interpretation': self._interpret_pp(pp_statistic, critical_values, p_value)
            }
            
            self.results['pp'] = test_result
            
            if self.verbose:
                self._print_pp_results(test_result)
            
            return test_result
        except Exception as e:
            if self.verbose:
                print(f"⚠️  Ошибка при выполнении теста Phillips-Perron: {e}")
            return None
    
    def _interpret_pp(self, statistic: float, critical_values: Dict, p_value: float) -> str:
        """
        Интерпретация результатов Phillips-Perron теста.
        
        Args:
            statistic: PP статистика
            critical_values: Критические значения
            p_value: p-value
            
        Returns:
            str: Интерпретация
        """
        if statistic < critical_values['1%']:
            strength = "сильно"
        elif statistic < critical_values['5%']:
            strength = "умеренно"
        elif statistic < critical_values['10%']:
            strength = "слабо"
        else:
            strength = "не"
        
        if p_value <= self.alpha:
            return f"Ряд стационарен ({strength} стационарен, p-value={p_value:.4f} <= {self.alpha})"
        else:
            return f"Ряд нестационарен (p-value={p_value:.4f} > {self.alpha})"
    
    def run_all_tests(self) -> Dict:
        """
        Запуск всех доступных тестов стационарности.
        
        Returns:
            dict: Сводка результатов всех тестов
        """
        if self.verbose:
            print("=" * 70)
            print("ТЕСТИРОВАНИЕ СТАЦИОНАРНОСТИ ВРЕМЕННОГО РЯДА")
            print("=" * 70)
            print(f"Количество наблюдений: {len(self.data)}")
            print(f"Уровень значимости: {self.alpha}")
            print("=" * 70)
            print()
        
        # Запускаем все тесты
        self.adf_test()
        if self.verbose:
            print()
        self.kpss_test()
        if self.verbose:
            print()
        pp_result = self.pp_test()
        if self.verbose and pp_result:
            print()
        
        # Сводка результатов
        summary = self.get_summary()
        
        if self.verbose:
            self._print_summary(summary)
        
        return summary
    
    def get_summary(self) -> Dict:
        """
        Получение сводки результатов всех тестов.
        
        Returns:
            dict: Сводка результатов
        """
        summary = {
            'n_obs': len(self.data),
            'alpha': self.alpha,
            'tests': {}
        }
        
        for test_name, test_result in self.results.items():
            if test_result:
                summary['tests'][test_name] = {
                    'is_stationary': test_result['is_stationary'],
                    'p_value': test_result['p_value'],
                    'statistic': test_result['statistic']
                }
        
        # Определяем общий вывод
        if len(summary['tests']) > 0:
            stationary_count = sum(1 for t in summary['tests'].values() if t['is_stationary'])
            total_tests = len(summary['tests'])
            
            summary['overall_conclusion'] = {
                'stationary_tests': stationary_count,
                'total_tests': total_tests,
                'is_stationary': stationary_count >= (total_tests / 2),  # Большинство тестов
                'recommendation': self._get_recommendation(summary['tests'])
            }
        
        return summary
    
    def _get_recommendation(self, tests: Dict) -> str:
        """
        Получение рекомендации на основе результатов тестов.
        
        Args:
            tests: Результаты тестов
            
        Returns:
            str: Рекомендация
        """
        adf_result = tests.get('adf', {})
        kpss_result = tests.get('kpss', {})
        
        if adf_result and kpss_result:
            adf_stationary = adf_result.get('is_stationary', False)
            kpss_stationary = kpss_result.get('is_stationary', False)
            
            if adf_stationary and kpss_stationary:
                return "Ряд стационарен. Оба теста подтверждают стационарность."
            elif adf_stationary and not kpss_stationary:
                return "Противоречивые результаты. ADF: стационарен, KPSS: нестационарен. Возможно, есть детерминированный тренд."
            elif not adf_stationary and kpss_stationary:
                return "Противоречивые результаты. ADF: нестационарен, KPSS: стационарен. Возможно, нужна дифференциация."
            else:
                return "Ряд нестационарен. Оба теста подтверждают нестационарность. Рекомендуется дифференциация."
        
        return "Недостаточно данных для рекомендации."
    
    def _print_adf_results(self, result: Dict):
        """Вывод результатов ADF теста."""
        print(f"📊 {result['test_name']}")
        print("-" * 70)
        print(f"ADF статистика: {result['statistic']:.4f}")
        print(f"p-value: {result['p_value']:.4f}")
        print(f"Использовано лагов: {result['used_lag']}")
        print(f"Количество наблюдений: {result['n_obs']}")
        print("Критические значения:")
        for key, value in result['critical_values'].items():
            print(f"  {key}: {value:.4f}")
        print(f"\n✅ ВЫВОД: {result['interpretation']}")
    
    def _print_kpss_results(self, result: Dict):
        """Вывод результатов KPSS теста."""
        print(f"📊 {result['test_name']}")
        print("-" * 70)
        print(f"KPSS статистика: {result['statistic']:.4f}")
        print(f"p-value: {result['p_value']:.4f}")
        print(f"Использовано лагов: {result['used_lag']}")
        print("Критические значения:")
        for key, value in result['critical_values'].items():
            print(f"  {key}: {value:.4f}")
        print(f"\n✅ ВЫВОД: {result['interpretation']}")
    
    def _print_pp_results(self, result: Dict):
        """Вывод результатов Phillips-Perron теста."""
        print(f"📊 {result['test_name']}")
        print("-" * 70)
        print(f"PP статистика: {result['statistic']:.4f}")
        print(f"p-value: {result['p_value']:.4f}")
        print("Критические значения:")
        for key, value in result['critical_values'].items():
            print(f"  {key}: {value:.4f}")
        print(f"\n✅ ВЫВОД: {result['interpretation']}")
    
    def _print_summary(self, summary: Dict):
        """Вывод сводки результатов."""
        print("=" * 70)
        print("📋 СВОДКА РЕЗУЛЬТАТОВ")
        print("=" * 70)
        
        if 'overall_conclusion' in summary:
            conclusion = summary['overall_conclusion']
            print(f"Тестов выполнено: {conclusion['total_tests']}")
            print(f"Тестов подтвердили стационарность: {conclusion['stationary_tests']}")
            print(f"\n🎯 Общий вывод: {'Ряд стационарен' if conclusion['is_stationary'] else 'Ряд нестационарен'}")
            print(f"\n💡 Рекомендация: {conclusion['recommendation']}")
        
        print("=" * 70)


# Удобная функция для быстрого использования
def test_stationarity(data: Union[np.ndarray, pd.Series, list],
                     alpha: float = 0.05,
                     tests: list = ['adf', 'kpss'],
                     verbose: bool = True) -> Dict:
    """
    Быстрый тест стационарности временного ряда.
    
    Args:
        data: Временной ряд для тестирования
        alpha: Уровень значимости (по умолчанию 0.05)
        tests: Список тестов для выполнения ['adf', 'kpss', 'pp']
        verbose: Выводить ли подробную информацию
        
    Returns:
        dict: Результаты тестов
        
    Example:
        >>> import numpy as np
        >>> data = np.random.randn(100)
        >>> results = test_stationarity(data)
    """
    tester = StationarityTest(data, alpha=alpha, verbose=verbose)
    
    if 'adf' in tests:
        tester.adf_test()
    if 'kpss' in tests:
        tester.kpss_test()
    if 'pp' in tests:
        tester.pp_test()
    
    return tester.get_summary()


# Пример использования (для обратной совместимости со старым кодом)
if __name__ == "__main__":
    # Пример 1: Стационарный ряд (белый шум)
    print("Пример 1: Стационарный ряд (белый шум)")
    print("=" * 70)
    np.random.seed(42)
    stationary_data = np.random.randn(100)
    test_stationarity(stationary_data)
    print("\n\n")
    
    # Пример 2: Нестационарный ряд (случайное блуждание)
    print("Пример 2: Нестационарный ряд (случайное блуждание)")
    print("=" * 70)
    non_stationary_data = np.cumsum(np.random.randn(100))
    test_stationarity(non_stationary_data)
    print("\n\n")
    
    # Пример 3: Использование класса напрямую
    print("Пример 3: Использование класса StationarityTest")
    print("=" * 70)
    tester = StationarityTest(stationary_data, alpha=0.05, verbose=True)
    tester.run_all_tests()
