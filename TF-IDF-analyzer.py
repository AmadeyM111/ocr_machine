# vectorization important documents by TF-IDF analyzer

from sklearn.feature_extraction.text import TfidfVectorizer
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

tfidf = TfidfVectorizer()
corpus_tfidf = tfidf.fit_transform(corpus)
corpus_tfidf 

tfidf_df = pd.DataFrame(
    corpus_tfidf.toarray(),columns=tfidf.get_feature_names_out(), index=corpus
)
tfidf_df

tfidf1 = TfidfVectorizer(min_df=2)
corpus_tfidf1 = tfidf1.fit_transform(corpus)
corpus_tfidf1

tfidf1_df = pd.DataFrame(
    corpus_tfidf1.toarray(), columns=tfidf1.get_feature_names_out(), index=corpus
)
tfidf1_df

print(tfidf1_df)