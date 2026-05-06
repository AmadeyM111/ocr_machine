# split text into tokens

import nltk
from nltk.tokenize import word_tokenize
from IPython.display import clear_output

nltk.download("punkt_tab")
clear_output()

text = "Do you want to enroll in the finest courses in machine learning?? \
i studied amazing courses in machine learning and i want to enroll in the finest courses in machine learning."
tokenized_text = word_tokenize(text)
print(f"source text: {text}")
print(f"Tokenized text: {tokenized_text}")

lowercased_text = [word.lower() for word in tokenized_text if word.isalpha()] # remove non-alphabetic characters
print(f"Tokenized text: {lowercased_text}")
print(f"Lowercased text: {tokenized_text}")

#  remove stopwords
from nltk.corpus import stopwords

nltk.download("stopwords")
clear_output()

print(f"stopwords in English: {stopwords.words('english')}")


STOPWORDS = set(stopwords.words('english'))
text_without_stopwords = [word for word in lowercased_text if word not in STOPWORDS]
print(f"Lowercased text: {lowercased_text}")
print(f"Text without stopwords: {text_without_stopwords}")

# Stemming and Lemming

from nltk import PorterStemmer

stemmer = PorterStemmer()
stemmed_text = [stemmer.stem(word) for word in text_without_stopwords]
print(f"Text without stopwords: {text_without_stopwords}")
print(f"Stemmed text: {stemmed_text}")

from nltk import WordNetLemmatizer
import nltk

nltk.download("wordnet")
try:
    from IPython.display import clear_output
    clear_output()
except ImportError:
    pass  # clear_output доступен только в Jupyter/IPython

nltk.download("averaged_perceptron_tagger_eng")
clear_output()

nltk.pos_tag(text_without_stopwords)

def pos_mapping(text):
    word_tag_array = []
    for word_tag in nltk.pos_tag(text):
        tag = word_tag[1][0]
        tag_dict = {"J": "a", "N": "n", "V": "v", "R": "r"}
        word_tag_array.append({word_tag[0], tag_dict[tag]})
    return word_tag_array

pos_mapping(text_without_stopwords)

text_without_stopwords = ['want', 'enroll', 'finest', 'courses', 'machine', 'learning', 'studied', 'amaizing']
lemmatized_text = ['want', 'enroll', 'finest', 'courses', 'machine', 'learning', 'studied', 'amaizing']
#16:14 check

final_text = " ".join(lemmatized_text)
print(f"Lemmatized text: {lemmatized_text}")
print(f"Final text: {final_text}")

lemmatizer = WordNetLemmatizer()
lemmatized_text = [lemmatizer.lemmatize(word) for word in text_without_stopwords]
print(f"Text without stopwords: {text_without_stopwords}")
print(f"Lemmatized text: {lemmatized_text}")

print(f'Lemma for word "studied" is {lemmatizer.lemmatize("studied", "v")}.')
print(f'Lemma for word "finest" is {lemmatizer.lemmatize("finest", "a")}.')

# 2 way to vectorize sentences 

import pandas as pd

corpus = pd.Series(
    [
       "Who then is free? The wise man who can govern himself.",
       "Rule your mind or it will rule you.",
       "Begin, be bold, and venture to be wise.",
       "Wisdom is not wisdom when it is derived from books alone.",
       "Anger is a brief madness.",
    ]
)

# Vectorization by Bag of Words method

from sklearn.feature_extraction.text import CountVectorizer

bow = CountVectorizer()
corpus_bow = bow.fit_transform(corpus)
corpus_bow

bow_df = pd.DataFrame(
    corpus_bow.toarray(), columns=bow.get_feature_names_out(), index=corpus
)
bow_df


bow1 = CountVectorizer(ngram_range=(2, 3))
corpus_bow1 = bow1.fit_transform(corpus)
corpus_bow1

bow1_df = pd.DataFrame(
    corpus_bow1.toarray(), columns=bow1.get_feature_names_out(), index=corpus
)
bow1_df

# analayzer parameter
bow2 = CountVectorizer(ngram_range=(4, 6), analyzer="char_wb")
corpus_bow2 = bow2.fit_transform(corpus)
corpus_bow2

bow2_df = pd.DataFrame(
    corpus_bow2.toarray(), columns=bow2.get_feature_names_out(), index=corpus
)
bow2_df

# vectorization of train and test data

from sklearn.feature_extraction.text import TfidfVectorizer

vect = TfidfVectorizer()
x_trian_vect = vect.fit_transform(x_train)
x_test_vect = vect.transform(x_test)
x_train_vect, x_test_vect

# Create model for classification Naive Bayes Classifire

from sklearn.naive_bayes import MultinomialNB

mnb = MultinomialNB()
mnb.fit(x_train_vect, y_train)
y_mnb = mnb.predict(x_test_vect)
y_mnb 

# Elevate classification quality by accuracy, precision_score, recall_score, f1_score

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

report = pd.DataFrame(
    [
        accuracy_score(y_test, y_mnb),
        recall_score(y_test, y_mnb),
        precision_score(y_test, y_mnb),
        f1_score(y_test, y_mmb),
    ]
).round(2)
report = report.rename(columns={0: "Naive Bayes"})
report = report.rename(
    index={0: "accuracy", 1: "recall", 2: "precision", 3: "f1-score", 4: "accuracy"}
)
report

import seaborn as sns
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_mnb)
plt.figure(figsize=(8, 6))

sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1], yticklabels=[1, 0]
)
plt.xlabel("Predicted labels")
plt.ylabel("True labels")
plt.show()