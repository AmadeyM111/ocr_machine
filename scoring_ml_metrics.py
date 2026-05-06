import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

# 1. Генерируем "фейковые" данные для примера
# (В реальности здесь будет загрузка вашего файла)
dummy_texts = [
    "free offer call now", "win lottery money", "hello friend how are you",
    "meeting at work tomorrow", "click this link for prize", "project deadline is soon",
    "buy cheap meds", "lunch at the cafeteria", "verify your bank account", 
    "let's go for a walk", "urgent transfer needed", "happy birthday to you"
]
dummy_labels = [1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1 - спам, 0 - не спам

# Превращаем в Series (не обязательно, но удобно)
X = pd.Series(dummy_texts)
y = pd.Series(dummy_labels)

# 2. Разбиваем на train (обучение) и test (проверку)
# test_size=0.2 означает, что 20% данных уйдет на проверку
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

pipeline = Pipeline(
    [
        ("vect", TfidfVectorizer()),
        ("clf", MultinomialNB()),
    ],
)
parameters = {
    "vect__ngram_range": ((1, 1), (1, 2)),
    "vect__min_df": (0.0001, 0.001),
    "vect__max_df": (0.7, 1.0),
}

grid_search = GridSearchCV(
    pipeline, parameters, n_jobs=-1, verbose=1, cv=5, scoring="f1"
)

grid_search.fit(x_train, y_train)
print(f"Best params: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.2f}")

# TFIDF split

pipeline = Pipeline(
    [
        (
            "vect",
            TfidfVectorizer(
                max_df=grid_search.best_params_["vect__max_df"],
                min_df=grid_search.best_params_["vect__min_df"],
                ngram_range=grid_search.best_params_["vect__ngram_range"],
            ),
        ),
        ("clf", MultinomialNB()),
    ],
)
pipeline.fit(x_train, y_train)
y_pred = pipeline.predict(x_test)

# report of metrics

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
from sklearn.metrics import confusion_matrix


report = pd.DataFrame(
    index=["Accuracy", "Recall", "Precision", "F1 Score"]
).round(2)

print(report)

# Lets compare how the predictions have changed by confusion matrix

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1], yticklabels=[1, 0]
)
plt.xlabel("Predicted labels")
plt.ylabel("True labels")
plt.show()