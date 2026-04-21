"""
Продвинутый OCR с предобработкой и пост-обработкой для минимизации CER.
"""

import logging
import re
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Union, List, Tuple, Dict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import json
from difflib import SequenceMatcher
import pytesseract
from pdf2image import convert_from_path
from PIL import Image, ImageFilter, ImageEnhance
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class AdvancedOCRConfig(BaseSettings):
    """Расширенная конфигурация с параметрами предобработки."""
    
    # Tesseract
    tesseract_cmd: str = "tesseract"
    default_lang: str = "rus+eng"
    oem: int = 3
    base_psm: int = 3
    timeout_sec: int = 300
    
    # Изображение
    dpi: int = 400
    color_space: str = "gray"
    
    # Предобработка OpenCV
    enable_preprocessing: bool = True
    denoise_strength: int = 10
    adaptive_block_size: int = 11
    adaptive_c: int = 2
    clahe_clip_limit: float = 2.0
    clahe_grid_size: int = 8
    min_contour_area: int = 50
    deskew: bool = True
    deskew_max_angle: float = 5.0
    
    # Пост-обработка
    enable_postprocessing: bool = True
    use_dictionary: bool = False
    dictionary_path: Optional[str] = None
    correct_common_errors: bool = True
    merge_hyphenated_words: bool = True
    
    # Параллелизация
    num_workers: int = -1
    cache_enabled: bool = False
    cache_dir: str = ".ocr_cache"
    
    model_config = SettingsConfigDict(env_file=".env", env_prefix="OCR_")


@dataclass
class ImageQualityMetrics:
    """Метрики качества изображения."""
    brightness: float
    contrast: float
    sharpness: float
    noise_level: float
    skew_angle: float
    text_density: float


class ImagePreprocessor:
    """Предобработка изображений для улучшения OCR."""
    
    def __init__(self, config: AdvancedOCRConfig):
        self.config = config
    
    def calculate_metrics(self, image: np.ndarray) -> ImageQualityMetrics:
        """Расчет метрик качества изображения."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        
        brightness = np.mean(gray) / 255.0
        contrast = np.std(gray) / 255.0
        
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = laplacian.var()
        
        median = cv2.medianBlur(gray, 3)
        noise_level = np.mean(cv2.absdiff(gray, median)) / 255.0
        
        skew_angle = self._estimate_skew_angle(gray)
        
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        text_density = np.sum(binary > 0) / binary.size
        
        return ImageQualityMetrics(
            brightness=brightness,
            contrast=contrast,
            sharpness=sharpness,
            noise_level=noise_level,
            skew_angle=skew_angle,
            text_density=text_density
        )
    
    def _estimate_skew_angle(self, image: np.ndarray) -> float:
        """Оценка угла наклона текста."""
        if not self.config.deskew:
            return 0.0
        
        edges = cv2.Canny(image, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
        
        if lines is None:
            return 0.0
        
        angles = []
        for rho, theta in (line[0] for line in lines[:20]):
            angle = (theta * 180 / np.pi) - 90
            if abs(angle) < self.config.deskew_max_angle:
                angles.append(angle)
        
        return np.median(angles) if angles else 0.0
    
    def preprocess(self, image: Image.Image) -> np.ndarray:
        """Полный пайплайн предобработки."""
        if not self.config.enable_preprocessing:
            return np.array(image)
        
        img = np.array(image)
        
        if len(img.shape) == 3:
            if self.config.color_space == "lab":
                lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
                gray = lab[:, :, 0]
            else:
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img
        
        if self.config.denoise_strength > 0:
            gray = cv2.fastNlMeansDenoising(gray, h=self.config.denoise_strength)
        
        if self.config.deskew:
            gray = self._deskew_image(gray)
        
        clahe = cv2.createCLAHE(
            clipLimit=self.config.clahe_clip_limit,
            tileGridSize=(self.config.clahe_grid_size, self.config.clahe_grid_size)
        )
        gray = clahe.apply(gray)
        
        binary = self._adaptive_threshold(gray)
        binary = self._morphological_operations(binary)
        binary = self._remove_small_objects(binary)
        
        return Image.fromarray(binary)
    
    def _deskew_image(self, image: np.ndarray) -> np.ndarray:
        """Коррекция наклона изображения."""
        angle = self._estimate_skew_angle(image)
        
        if abs(angle) < 0.1:
            return image
        
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        cos = np.abs(matrix[0, 0])
        sin = np.abs(matrix[0, 1])
        
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        
        matrix[0, 2] += (new_w / 2) - center[0]
        matrix[1, 2] += (new_h / 2) - center[1]
        
        return cv2.warpAffine(image, matrix, (new_w, new_h), 
                             flags=cv2.INTER_CUBIC, 
                             borderMode=cv2.BORDER_REPLICATE)
    
    def _adaptive_threshold(self, image: np.ndarray) -> np.ndarray:
        """Адаптивная бинаризация."""
        _, otsu = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        adaptive = cv2.adaptiveThreshold(
            image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            self.config.adaptive_block_size,
            self.config.adaptive_c
        )
        
        metrics = self.calculate_metrics(image)
        
        if metrics.contrast < 0.3 or metrics.brightness < 0.3 or metrics.brightness > 0.7:
            return adaptive
        else:
            return otsu
    
    def _morphological_operations(self, image: np.ndarray) -> np.ndarray:
        """Морфологические операции."""
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=1)
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)
        return opened
    
    def _remove_small_objects(self, image: np.ndarray) -> np.ndarray:
        """Удаление мелких объектов."""
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            image, connectivity=8
        )
        
        mask = np.zeros_like(image)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= self.config.min_contour_area:
                mask[labels == i] = 255
        
        return mask


class TextPostprocessor:
    """Пост-обработка распознанного текста."""
    
    def __init__(self, config: AdvancedOCRConfig):
        self.config = config
        self.dictionary = self._load_dictionary()
    
    def _load_dictionary(self) -> set:
        """Загрузка словаря."""
        if not self.config.use_dictionary or not self.config.dictionary_path:
            return set()
        
        try:
            with open(self.config.dictionary_path, 'r', encoding='utf-8') as f:
                return set(word.strip().lower() for word in f)
        except Exception as e:
            logger.warning(f"Не удалось загрузить словарь: {e}")
            return set()
    
    def postprocess(self, text: str) -> str:
        """Полный пайплайн пост-обработки."""
        if not self.config.enable_postprocessing:
            return text
        
        text = self._normalize_whitespace(text)
        
        if self.config.correct_common_errors:
            text = self._correct_common_errors(text)
        
        if self.config.merge_hyphenated_words:
            text = self._merge_hyphenated_words(text)
        
        if self.dictionary:
            text = self._spell_check(text)
        
        text = self._remove_artifacts(text)
        
        return text
    
    def _normalize_whitespace(self, text: str) -> str:
        """Нормализация пробелов."""
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r' +([,.!?;:])', r'\1', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        return text.strip()
    
    def _correct_common_errors(self, text: str) -> str:
        """Исправление типичных ошибок OCR."""
        replacements = {
            r'\b0\b': 'о',
            r'\b1\b': 'l',
            r'[\u0401\u0451]': 'е',
            r'([а-я])\1{2,}': r'\1\1',
        }
        
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text
    
    def _merge_hyphenated_words(self, text: str) -> str:
        """Слияние слов с переносом."""
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
        return text
    
    def _spell_check(self, text: str) -> str:
        """Проверка орфографии."""
        words = re.findall(r'\b\w+\b', text)
        
        for word in words:
            if word.lower() not in self.dictionary:
                candidates = [
                    w for w in self.dictionary
                    if abs(len(w) - len(word)) <= 2 and 
                       SequenceMatcher(None, w, word.lower()).ratio() > 0.8
                ]
                if candidates:
                    best = max(candidates, key=lambda w: SequenceMatcher(None, w, word.lower()).ratio())
                    text = text.replace(word, best, 1)
        
        return text
    
    def _remove_artifacts(self, text: str) -> str:
        """Удаление артефактов."""
        text = re.sub(r'\b[^а-яА-Яa-zA-Z0-9]{1,2}\b', '', text)
        lines = text.split('\n')
        lines = [l for l in lines if re.search(r'[а-яА-Яa-zA-Z0-9]', l)]
        return '\n'.join(lines)


class AdvancedPDFTextExtractor:
    """Продвинутый экстрактор текста из PDF."""
    
    def __init__(self, config: Optional[AdvancedOCRConfig] = None):
        self.config = config or AdvancedOCRConfig()
        pytesseract.pytesseract.tesseract_cmd = self.config.tesseract_cmd
        
        self.preprocessor = ImagePreprocessor(self.config)
        self.postprocessor = TextPostprocessor(self.config)
        
        if self.config.cache_enabled:
            self.cache_dir = Path(self.config.cache_dir)
            self.cache_dir.mkdir(exist_ok=True)
        
        if self.config.num_workers == -1:
            import os
            self.config.num_workers = os.cpu_count() or 4
    
    def _process_single_page(
        self, 
        args: Tuple[int, Image.Image, str, int]
    ) -> Tuple[int, str, Dict]:
        """Обработка одной страницы."""
        page_num, image, lang, psm = args
        
        preprocessed = self.preprocessor.preprocess(image)
        
        config_str = f"--oem {self.config.oem} --psm {psm}"
        
        text = pytesseract.image_to_string(
            preprocessed,
            lang=lang,
            config=config_str,
            timeout=self.config.timeout_sec
        )
        
        text = self.postprocessor.postprocess(text)
        
        metrics = {
            'cached': False,
            'image_size': image.size,
        }
        
        return page_num, text, metrics
    
    def _select_psm(self, image: Image.Image) -> int:
        """Автовыбор PSM."""
        img_array = np.array(image)
        metrics = self.preprocessor.calculate_metrics(img_array)
        
        if metrics.text_density < 0.05:
            return 11
        elif metrics.text_density > 0.5:
            return 6
        else:
            return 3
    
    def extract_text(
        self,
        pdf_path: Union[str, Path],
        lang: Optional[str] = None,
        force_ocr: bool = False,
        use_parallel: bool = True
    ) -> str:
        """Извлечение текста из PDF."""
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"Файл не найден: {pdf_path}")
        
        target_lang = lang or self.config.default_lang
        logger.info(f"Обработка: {pdf_path.name}, язык: {target_lang}")
        
        convert_kwargs = {"dpi": self.config.dpi}
        
        images = convert_from_path(str(pdf_path), **convert_kwargs)
        logger.info(f"Конвертировано страниц: {len(images)}")
        
        tasks = []
        for idx, img in enumerate(images, 1):
            psm = self._select_psm(img) if self.config.base_psm == 3 else self.config.base_psm
            tasks.append((idx, img, target_lang, psm))
        
        results = {}
        
        if use_parallel and len(tasks) > 1 and self.config.num_workers > 1:
            with ProcessPoolExecutor(max_workers=self.config.num_workers) as executor:
                futures = {executor.submit(self._process_single_page, t): t[0] for t in tasks}
                
                for future in as_completed(futures):
                    page_num, text, metrics = future.result()
                    results[page_num] = text
        else:
            for task in tasks:
                page_num, text, metrics = self._process_single_page(task)
                results[page_num] = text
        
        pages_text = [results[i] for i in sorted(results.keys())]
        result = "\n\n---\n\n".join(pages_text)
        
        logger.info("OCR завершен успешно")
        return result


def extract_text_advanced(
    pdf_path: str,
    lang: str = "rus+eng",
    dpi: int = 400,
    enable_preprocessing: bool = True,
    enable_postprocessing: bool = True,
    num_workers: int = -1
) -> str:
    """Быстрая функция для извлечения текста."""
    config = AdvancedOCRConfig(
        default_lang=lang,
        dpi=dpi,
        enable_preprocessing=enable_preprocessing,
        enable_postprocessing=enable_postprocessing,
        num_workers=num_workers
    )
    extractor = AdvancedPDFTextExtractor(config)
    return extractor.extract_text(pdf_path)