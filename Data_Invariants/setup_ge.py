import great_expectations as gx
import os
import logging
from pathlib import Path
from typing import Optional, Any

# Попытка загрузить переменные окружения из .env файла (опционально)
try:
    from dotenv import load_dotenv
    # Загружаем переменные из .env файла, если он существует
    env_path = Path(__file__).parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        logging.getLogger(__name__).debug(f"Переменные окружения загружены из {env_path}")
except ImportError:
    # python-dotenv не установлен - используем только системные переменные окружения
    pass

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Определяем корень проекта относительно расположения этого файла
# Скрипт находится в: Active/experimental/Data_Invariants/
# Данные находятся в: Active/experimental/data/
# Используем абсолютный путь для надежности
try:
    # Получаем директорию, где находится скрипт
    PROJECT_ROOT = Path(__file__).parent.resolve()
except NameError:
    # Если __file__ недоступен (например, в интерактивном режиме), используем текущую директорию
    PROJECT_ROOT = Path.cwd().resolve()


def get_config_from_env() -> dict:
    """
    Получение конфигурации из переменных окружения.
    Возвращает словарь с настройками или значения по умолчанию.
    
    Пути по умолчанию задаются относительно корня проекта для лучшей переносимости.
    """
    # Относительные пути от каталога запуска скрипта
    # Скрипт находится в: Active/experimental/Data_Invariants/
    # Данные находятся в: Active/experimental/data/ (на уровень выше от Data_Invariants)
    
    # Логируем для отладки (используем INFO для видимости)
    logger.info(f"PROJECT_ROOT: {PROJECT_ROOT}")
    logger.info(f"PROJECT_ROOT.parent: {PROJECT_ROOT.parent}")
    
    # Вычисляем пути напрямую, без дополнительных resolve()
    # PROJECT_ROOT уже является абсолютным путем после resolve() в определении
    parent_dir = PROJECT_ROOT.parent
    default_raw_data = parent_dir / 'data' / 'raw'
    default_clean_data = parent_dir / 'data' / 'processed'
    
    logger.info(f"Вычисленный путь к raw данным (до resolve): {default_raw_data}")
    logger.info(f"Вычисленный путь к processed данным (до resolve): {default_clean_data}")
    
    # Преобразуем в абсолютные пути, но аккуратно
    default_raw_data = default_raw_data.resolve()
    default_clean_data = default_clean_data.resolve()
    
    logger.info(f"Вычисленный путь к raw данным (после resolve): {default_raw_data}")
    logger.info(f"Вычисленный путь к processed данным (после resolve): {default_clean_data}")
    
    # Получаем пути из переменных окружения или используем относительные по умолчанию
    raw_data_path = os.getenv('GE_RAW_DATA_PATH')
    if raw_data_path:
        # Если путь задан через переменную окружения, преобразуем его в абсолютный
        # Если путь относительный, разрешаем его относительно PROJECT_ROOT
        path_obj = Path(raw_data_path).expanduser()
        if not path_obj.is_absolute():
            raw_data_path = str((PROJECT_ROOT / path_obj).resolve())
        else:
            raw_data_path = str(path_obj.resolve())
        logger.info(f"Путь из переменной окружения GE_RAW_DATA_PATH: {raw_data_path}")
    else:
        # Используем относительный путь от корня проекта
        # default_raw_data уже является абсолютным путем после resolve()
        raw_data_path = str(default_raw_data)
        logger.info(f"Используется путь по умолчанию: {raw_data_path}")
    
    clean_data_path = os.getenv('GE_CLEAN_DATA_PATH')
    if clean_data_path:
        # Если путь задан через переменную окружения, преобразуем его в абсолютный
        # Если путь относительный, разрешаем его относительно PROJECT_ROOT
        path_obj = Path(clean_data_path).expanduser()
        if not path_obj.is_absolute():
            clean_data_path = str((PROJECT_ROOT / path_obj).resolve())
        else:
            clean_data_path = str(path_obj.resolve())
        logger.info(f"Путь к processed данным из переменной окружения GE_CLEAN_DATA_PATH: {clean_data_path}")
    else:
        # Используем относительный путь от корня проекта
        # default_clean_data уже является абсолютным путем после resolve()
        clean_data_path = str(default_clean_data)
        logger.info(f"Используется путь к processed данным по умолчанию: {clean_data_path}")
    
    # Паттерн для поиска файлов (поддерживает glob patterns для работы с поддиректориями)
    # Примеры: "**/*.json" - все JSON файлы рекурсивно
    #          "class*/codec*/*.json" - файлы в поддиректориях по классам и кодексам
    file_pattern = os.getenv('GE_FILE_PATTERN', '**/*.json')
    
    config = {
        'raw_data_path': raw_data_path,
        'clean_data_path': clean_data_path,
        'data_source_name': os.getenv('GE_DATA_SOURCE_NAME', 'my_local_datasource'),
        'asset_name': os.getenv('GE_ASSET_NAME', 'legal_json_asset'),
        'batch_definition_name': os.getenv('GE_BATCH_DEFINITION_NAME', 'whole_file'),
        'file_pattern': file_pattern,  # Паттерн для поиска файлов в поддиректориях
    }
    logger.info(f"Конфигурация загружена: {config}")
    return config


def validate_paths(config: dict) -> None:
    """
    Валидация существования директорий и файлов по паттерну.
    
    Args:
        config: Словарь с конфигурацией
        
    Raises:
        FileNotFoundError: Если директория не существует или файлы не найдены
    """
    raw_data_path = Path(config['raw_data_path'])
    if not raw_data_path.exists():
        error_msg = f"Директория raw_data не найдена: {raw_data_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    logger.info(f"Директория raw_data найдена: {raw_data_path}")
    
    # Проверяем наличие файлов по паттерну
    file_pattern = config.get('file_pattern', '**/*.json')
    matching_files = list(raw_data_path.glob(file_pattern))
    
    if not matching_files:
        error_msg = (
            f"Не найдено файлов по паттерну '{file_pattern}' в директории: {raw_data_path}\n"
            f"Убедитесь, что:\n"
            f"  1. Путь GE_RAW_DATA_PATH указывает на директорию с данными\n"
            f"  2. В директории есть файлы, соответствующие паттерну '{file_pattern}'\n"
            f"  3. Текущий путь: {raw_data_path.absolute()}"
        )
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    logger.info(f"Найдено {len(matching_files)} файл(ов) по паттерну '{file_pattern}'")
    if len(matching_files) <= 10:
        logger.debug(f"Найденные файлы: {[str(f.relative_to(raw_data_path)) for f in matching_files]}")
    else:
        logger.debug(f"Первые 10 файлов: {[str(f.relative_to(raw_data_path)) for f in matching_files[:10]]}")


def setup_great_expectations(config: dict) -> tuple:
    """
    Настройка Great Expectations: создание контекста, источника данных и asset.
    
    Args:
        config: Словарь с конфигурацией
        
    Returns:
        tuple: (context, data_source, data_asset, batch_definition)
        
    Raises:
        Exception: При ошибках инициализации Great Expectations
    """
    try:
        logger.info("Инициализация контекста Great Expectations...")
        context = gx.get_context()
        logger.info("Контекст успешно инициализирован")
        
        logger.info(f"Создание источника данных: {config['data_source_name']}")
        data_source = context.data_sources.add_pandas_filesystem(
            name=config['data_source_name'],
            base_directory=config['raw_data_path']
        )
        logger.info(f"Источник данных '{config['data_source_name']}' успешно создан")
        
        logger.info(f"Создание Data Asset: {config['asset_name']}")
        logger.info(f"Использование паттерна файлов: {config['file_pattern']}")
        
        # Создаем Data Asset с паттерном для поиска файлов в поддиректориях
        # Это позволяет работать с данными, разбитыми по классам/кодексам
        # Пробуем разные варианты API для совместимости с разными версиями
        try:
            # Для новых версий Great Expectations
            data_asset = data_source.add_json_asset(
                name=config['asset_name'],
                glob_directive=config['file_pattern']
            )
            logger.info(f"Data Asset '{config['asset_name']}' успешно создан с паттерном '{config['file_pattern']}'")
        except TypeError:
            try:
                # Альтернативный вариант с batching_regex
                data_asset = data_source.add_json_asset(
                    name=config['asset_name'],
                    batching_regex=config['file_pattern']
                )
                logger.info(f"Data Asset '{config['asset_name']}' успешно создан с паттерном '{config['file_pattern']}'")
            except TypeError:
                # Если паттерн не поддерживается напрямую, создаем asset без паттерна
                # и настроим batch definition с паттерном позже
                logger.warning("Прямая поддержка паттернов не найдена, используем базовый asset")
                data_asset = data_source.add_json_asset(name=config['asset_name'])
        except Exception as e:
            # Обработка ошибки, когда файлы не найдены по паттерну
            error_msg = str(e)
            if "No file" in error_msg or "matched" in error_msg.lower():
                logger.error(
                    f"Не найдено файлов по паттерну '{config['file_pattern']}' в директории '{config['raw_data_path']}'\n"
                    f"Проверьте:\n"
                    f"  1. Правильность пути GE_RAW_DATA_PATH (текущий: {config['raw_data_path']})\n"
                    f"  2. Наличие файлов, соответствующих паттерну '{config['file_pattern']}'\n"
                    f"  3. Права доступа к директории и файлам"
                )
            raise
        
        logger.info(f"Создание Batch Definition: {config['batch_definition_name']}")
        # В новых версиях Great Expectations для JSON assets batch definitions создаются автоматически
        # или используются другие методы. Пробуем разные варианты API.
        batch_definition = None
        try:
            # Пробуем метод whole_file (если доступен)
            if hasattr(data_asset, 'add_batch_definition_whole_file'):
                batch_definition = data_asset.add_batch_definition_whole_file(
                    config['batch_definition_name']
                )
                logger.info(f"Batch Definition '{config['batch_definition_name']}' успешно создан")
            else:
                # Для новых версий: batch definitions создаются автоматически при создании asset
                # или можно использовать build_batch_request() напрямую
                logger.info("Batch Definition создается автоматически при использовании asset")
                # Создаем тестовый batch request для проверки
                try:
                    test_batch_request = data_asset.build_batch_request()
                    logger.info(f"Batch request успешно создан: {test_batch_request}")
                    batch_definition = test_batch_request  # Используем batch_request как batch_definition
                except Exception as e:
                    logger.warning(f"Не удалось создать batch request: {e}")
                    logger.info("Продолжаем без явного batch definition")
        except Exception as e:
            logger.warning(f"Ошибка при создании Batch Definition: {e}")
            logger.info("Продолжаем без явного batch definition - он будет создан автоматически")
        
        return context, data_source, data_asset, batch_definition
        
    except Exception as e:
        logger.error(f"Ошибка при настройке Great Expectations: {str(e)}", exc_info=True)
        raise


def create_expectations(context: Any, data_asset: Any) -> Optional[Any]:
    """
    Создание набора expectations для проверки данных.
    
    Args:
        context: Контекст Great Expectations
        data_asset: Data Asset для создания expectations
        
    Returns:
        ExpectationSuite или None при ошибке
    """
    try:
        logger.info("Создание Expectation Suite...")
        suite_name = f"{data_asset.name}_suite"
        
        # Получаем или создаем Expectation Suite
        suite = None
        suite_exists = False
        
        try:
            # Пытаемся получить существующий suite
            if hasattr(context.suites, 'get'):
                try:
                    suite = context.suites.get(suite_name)
                    suite_exists = True
                    logger.info(f"Использование существующего Expectation Suite: {suite_name}")
                except (KeyError, AttributeError):
                    # Suite не существует, создадим новый
                    suite_exists = False
                    logger.info(f"Expectation Suite '{suite_name}' не найден, будет создан новый")
            else:
                # Альтернативный способ проверки для разных версий API
                try:
                    # Пробуем получить suite через список
                    all_suites = list(context.suites.list_all())
                    if suite_name in all_suites:
                        suite = context.suites.get(suite_name)
                        suite_exists = True
                        logger.info(f"Использование существующего Expectation Suite: {suite_name}")
                    else:
                        suite_exists = False
                        logger.info(f"Expectation Suite '{suite_name}' не найден, будет создан новый")
                except Exception:
                    suite_exists = False
                    logger.info(f"Не удалось проверить существование suite, создадим новый")
        except Exception as e:
            logger.debug(f"Ошибка при проверке существования suite: {e}")
            suite_exists = False
        
        # Создаем новый suite только если он не существует
        if not suite_exists:
            try:
                suite = context.suites.add(gx.ExpectationSuite(name=suite_name))
                logger.info(f"Создан новый Expectation Suite: {suite_name}")
            except Exception as e:
                # Если suite уже существует (race condition или другая причина)
                if "already exists" in str(e).lower() or "exists" in str(e).lower():
                    logger.info(f"Expectation Suite '{suite_name}' уже существует, используем его")
                    try:
                        suite = context.suites.get(suite_name)
                        suite_exists = True
                    except Exception:
                        logger.warning(f"Не удалось получить существующий suite: {e}")
                        raise
                else:
                    raise
        
        # Получаем валидатор для работы с данными
        logger.info("Получение валидатора для создания expectations...")
        batch_request = data_asset.build_batch_request()
        
        # Получаем валидатор (разные версии API могут иметь разные методы)
        try:
            validator = context.get_validator(
                batch_request=batch_request,
                expectation_suite_name=suite_name
            )
        except AttributeError:
            # Альтернативный способ для старых версий
            validator = context.get_validator(
                batch_request=batch_request,
                expectation_suite=suite
            )
        
        # Добавляем expectations для проверки данных
        logger.info("Добавление expectations...")
        
        # Примеры expectations для JSON данных:
        # 1. Проверка, что данные не пустые
        try:
            validator.expect_table_row_count_to_be_between(min_value=1)
            logger.info("✓ Expectation добавлена: таблица не пустая")
        except Exception as e:
            logger.warning(f"Не удалось добавить expectation для проверки количества строк: {e}")
        
        # 2. Проверка наличия обязательных колонок (если данные структурированы)
        # Это будет работать только если JSON преобразован в таблицу
        # Пример для случая, когда JSON имеет структуру с колонками:
        # try:
        #     validator.expect_column_to_exist("column_name")
        #     logger.info("✓ Expectation добавлена: проверка наличия колонки")
        # except Exception as e:
        #     logger.debug(f"Проверка колонок не применима: {e}")
        
        # 3. Проверка уникальности (если применимо)
        # try:
        #     validator.expect_column_values_to_be_unique("id")
        #     logger.info("✓ Expectation добавлена: проверка уникальности")
        # except Exception as e:
        #     logger.debug(f"Проверка уникальности не применима: {e}")
        
        # Сохраняем suite
        # Если suite уже существовал, save_expectation_suite() обновит его
        # Если suite новый, save_expectation_suite() добавит его
        try:
            validator.save_expectation_suite()
            if suite_exists:
                logger.info(f"Expectation Suite '{suite_name}' успешно обновлен")
            else:
                logger.info(f"Expectation Suite '{suite_name}' успешно сохранен")
        except Exception as e:
            # Если suite уже существует при сохранении, это нормально - значит он был обновлен
            if "already exists" in str(e).lower() or "exists" in str(e).lower():
                logger.info(f"Expectation Suite '{suite_name}' уже существует, обновление выполнено через validator")
            else:
                # Другая ошибка - пробрасываем дальше
                raise
        
        # Получаем обновленный suite из validator, чтобы убедиться, что он содержит все expectations
        # Это важно, так как suite объект может не обновиться автоматически
        try:
            updated_suite = validator.get_expectation_suite()
            logger.info(f"Suite из validator содержит {len(updated_suite.expectations)} expectations")
            return updated_suite
        except Exception as e:
            logger.warning(f"Не удалось получить suite из validator: {e}, возвращаем исходный suite")
            # Если не удалось получить из validator, пробуем получить из контекста
            try:
                updated_suite = context.suites.get(suite_name)
                logger.info(f"Suite из контекста содержит {len(updated_suite.expectations)} expectations")
                return updated_suite
            except Exception:
                logger.warning("Не удалось получить suite из контекста, возвращаем исходный suite")
                return suite
        
    except Exception as e:
        logger.error(f"Ошибка при создании expectations: {str(e)}", exc_info=True)
        return None


def validate_data(context: Any, data_asset: Any, suite_name: str, suite: Optional[Any] = None) -> Optional[Any]:
    """
    Выполнение валидации данных с использованием созданных expectations.
    
    Args:
        context: Контекст Great Expectations
        data_asset: Data Asset для валидации
        suite_name: Имя Expectation Suite
        suite: Объект ExpectationSuite (опционально, если передан, используется напрямую)
        
    Returns:
        CheckpointResult или None при ошибке
    """
    try:
        logger.info("Запуск валидации данных...")
        
        # Получаем suite из контекста, если не передан напрямую
        if suite is None:
            try:
                suite = context.suites.get(suite_name)
                logger.info(f"Suite '{suite_name}' загружен из контекста, содержит {len(suite.expectations)} expectations")
            except Exception as e:
                logger.warning(f"Не удалось получить suite '{suite_name}' из контекста: {e}")
                logger.info("Продолжаем с suite_name при создании validator")
        else:
            # Если suite передан, проверяем, что он содержит expectations
            logger.info(f"Используется переданный suite объект, содержит {len(suite.expectations)} expectations")
            # Если suite пустой, пробуем перезагрузить из контекста
            if len(suite.expectations) == 0:
                logger.warning("Переданный suite пустой, пробуем перезагрузить из контекста")
                try:
                    suite = context.suites.get(suite_name)
                    logger.info(f"Suite перезагружен из контекста, содержит {len(suite.expectations)} expectations")
                except Exception as e:
                    logger.warning(f"Не удалось перезагрузить suite из контекста: {e}")
                    logger.info("Продолжаем с переданным suite")
        
        # Создаем checkpoint для валидации
        checkpoint_name = f"{data_asset.name}_checkpoint"
        
        batch_request = data_asset.build_batch_request()
        
        # Пытаемся получить существующий checkpoint или создать новый
        checkpoint = None
        checkpoint_exists = False
        
        try:
            # Пробуем получить существующий checkpoint
            if hasattr(context.checkpoints, 'get'):
                try:
                    checkpoint = context.checkpoints.get(checkpoint_name)
                    checkpoint_exists = True
                    logger.info(f"Использование существующего Checkpoint: {checkpoint_name}")
                except (KeyError, AttributeError, Exception) as e:
                    # Checkpoint не существует, создадим новый
                    checkpoint_exists = False
                    logger.info(f"Checkpoint '{checkpoint_name}' не найден, будет создан новый")
            else:
                # Альтернативный способ проверки для разных версий API
                try:
                    all_checkpoints = list(context.checkpoints.list_all())
                    if checkpoint_name in all_checkpoints:
                        checkpoint = context.checkpoints.get(checkpoint_name)
                        checkpoint_exists = True
                        logger.info(f"Использование существующего Checkpoint: {checkpoint_name}")
                    else:
                        checkpoint_exists = False
                        logger.info(f"Checkpoint '{checkpoint_name}' не найден, будет создан новый")
                except Exception:
                    checkpoint_exists = False
                    logger.info(f"Не удалось проверить существование checkpoint, создадим новый")
        except Exception as e:
            logger.debug(f"Ошибка при проверке существования checkpoint: {e}")
            checkpoint_exists = False
        
        # Создаем новый checkpoint только если он не существует
        if not checkpoint_exists:
            try:
                # Пробуем разные способы создания checkpoint в зависимости от версии Great Expectations
                # Способ 1: SimpleCheckpoint (для новых версий)
                try:
                    from great_expectations.checkpoint import SimpleCheckpoint
                    
                    checkpoint = SimpleCheckpoint(
                        name=checkpoint_name,
                        data_context=context,
                        validations=[
                            {
                                "batch_request": batch_request,
                                "expectation_suite_name": suite_name,
                            }
                        ],
                    )
                    # Добавляем checkpoint в контекст
                    context.checkpoints.add(checkpoint)
                    logger.info(f"Checkpoint '{checkpoint_name}' успешно создан через SimpleCheckpoint")
                except (ImportError, TypeError, AttributeError) as e1:
                    # Способ 2: Прямое создание через checkpoint.run() без сохранения checkpoint
                    logger.info(f"SimpleCheckpoint недоступен или не работает ({e1}), используем валидацию через validator")
                    
                    # Создаем validator с явным указанием suite объекта, если доступен
                    try:
                        if suite is not None:
                            validator = context.get_validator(
                                batch_request=batch_request,
                                expectation_suite=suite
                            )
                            logger.info(f"Validator создан с suite объектом, содержит {len(suite.expectations)} expectations")
                        else:
                            validator = context.get_validator(
                                batch_request=batch_request,
                                expectation_suite_name=suite_name
                            )
                            validator_suite = validator.get_expectation_suite()
                            logger.info(f"Validator создан с suite_name, suite содержит {len(validator_suite.expectations)} expectations")
                    except Exception as validator_error:
                        logger.warning(f"Ошибка при создании validator с suite объектом: {validator_error}, пробуем с suite_name")
                        validator = context.get_validator(
                            batch_request=batch_request,
                            expectation_suite_name=suite_name
                        )
                        validator_suite = validator.get_expectation_suite()
                        logger.info(f"Validator создан с suite_name, suite содержит {len(validator_suite.expectations)} expectations")
                    
                    result = validator.validate()
                    logger.info("✓ Валидация выполнена напрямую через validator")
                    return result
            except Exception as e:
                # Если checkpoint уже существует (race condition или другая причина)
                if "already exists" in str(e).lower() or "exists" in str(e).lower():
                    logger.info(f"Checkpoint '{checkpoint_name}' уже существует, используем его")
                    try:
                        checkpoint = context.checkpoints.get(checkpoint_name)
                    except Exception:
                        logger.warning(f"Не удалось получить существующий checkpoint: {e}")
                        # Пробуем валидацию напрямую
                        logger.info("Пробуем выполнить валидацию напрямую через validator...")
                        
                        # Создаем validator с явным указанием suite объекта, если доступен
                        try:
                            if suite is not None:
                                validator = context.get_validator(
                                    batch_request=batch_request,
                                    expectation_suite=suite
                                )
                                logger.info(f"Validator создан с suite объектом, содержит {len(suite.expectations)} expectations")
                            else:
                                validator = context.get_validator(
                                    batch_request=batch_request,
                                    expectation_suite_name=suite_name
                                )
                                validator_suite = validator.get_expectation_suite()
                                logger.info(f"Validator создан с suite_name, suite содержит {len(validator_suite.expectations)} expectations")
                        except Exception as validator_error:
                            logger.warning(f"Ошибка при создании validator: {validator_error}")
                            validator = context.get_validator(
                                batch_request=batch_request,
                                expectation_suite_name=suite_name
                            )
                            validator_suite = validator.get_expectation_suite()
                            logger.info(f"Validator создан с suite_name, suite содержит {len(validator_suite.expectations)} expectations")
                        
                        result = validator.validate()
                        logger.info("✓ Валидация выполнена напрямую через validator")
                        return result
                else:
                    # Пробуем альтернативный способ - валидацию через validator напрямую
                    logger.warning(f"Не удалось создать checkpoint: {e}")
                    logger.info("Пробуем выполнить валидацию напрямую через validator...")
                    try:
                        # Создаем validator с явным указанием suite объекта, если доступен
                        try:
                            if suite is not None:
                                validator = context.get_validator(
                                    batch_request=batch_request,
                                    expectation_suite=suite
                                )
                                logger.info(f"Validator создан с suite объектом, содержит {len(suite.expectations)} expectations")
                            else:
                                validator = context.get_validator(
                                    batch_request=batch_request,
                                    expectation_suite_name=suite_name
                                )
                                validator_suite = validator.get_expectation_suite()
                                logger.info(f"Validator создан с suite_name, suite содержит {len(validator_suite.expectations)} expectations")
                        except Exception as validator_error:
                            logger.warning(f"Ошибка при создании validator: {validator_error}")
                            validator = context.get_validator(
                                batch_request=batch_request,
                                expectation_suite_name=suite_name
                            )
                            validator_suite = validator.get_expectation_suite()
                            logger.info(f"Validator создан с suite_name, suite содержит {len(validator_suite.expectations)} expectations")
                        
                        result = validator.validate()
                        logger.info("✓ Валидация выполнена напрямую через validator")
                        return result
                    except Exception as e2:
                        logger.error(f"Ошибка при прямой валидации: {e2}")
                        raise
        
        # Запускаем валидацию
        logger.info("Выполнение валидации...")
        result = checkpoint.run()
        
        # Проверяем результат
        if hasattr(result, 'success'):
            if result.success:
                logger.info("✓ Валидация прошла успешно!")
            else:
                logger.warning("⚠ Валидация завершилась с предупреждениями или ошибками")
        else:
            # Для некоторых версий API результат может иметь другую структуру
            logger.info("Валидация выполнена")
        
        logger.info(f"Результаты валидации: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Ошибка при валидации данных: {str(e)}", exc_info=True)
        return None


def main():
    """
    Главная функция для запуска всего процесса настройки и валидации данных.
    """
    try:
        logger.info("=" * 60)
        logger.info("Запуск процесса настройки Great Expectations")
        logger.info("=" * 60)
        
        # 1. Загрузка конфигурации из переменных окружения
        config = get_config_from_env()
        
        # 2. Валидация путей
        validate_paths(config)
        
        # 3. Настройка Great Expectations
        context, data_source, data_asset, batch_definition = setup_great_expectations(config)
        
        # 4. Создание expectations
        suite = create_expectations(context, data_asset)
        
        # 5. Выполнение валидации данных (если suite создан успешно)
        if suite:
            suite_name = suite.name
            validation_result = validate_data(context, data_asset, suite_name, suite=suite)
            
            if validation_result:
                logger.info("=" * 60)
                logger.info("Процесс валидации завершен успешно")
                logger.info("=" * 60)
            else:
                logger.warning("Валидация не была выполнена из-за ошибок")
        else:
            logger.warning("Expectations не были созданы, валидация пропущена")
        
    except FileNotFoundError as e:
        logger.error(f"Ошибка файловой системы: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Критическая ошибка: {str(e)}", exc_info=True)
        raise


# DEFINITIONS:
# Context: Мозг системы (хранит настройки).
# Data Source: Указывает на папку с данными.
# Data Asset: Указывает на конкретный тип файла в этой папке.
# Batch: Конкретные данные, которые будут проверяться тестами.
# Expectation Suite: Набор правил для проверки данных.
# Validator: Объект для выполнения проверок на данных.
# Checkpoint: Точка проверки, объединяющая данные и expectations.

if __name__ == "__main__":
    main()
