from nltk.corptus import stopwords

nltk.download("stopwords")
clear_output()

print(f"Stopwords in English: {stopwords.words('english)}")