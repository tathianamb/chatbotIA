from nltk.tokenize import word_tokenize
import re
from nltk.corpus import stopwords
import spacy

nlp = spacy.load("pt_core_news_sm")
stop_words = set(stopwords.words('portuguese'))


def process_text(text):
    words = word_tokenize(text.lower(), language="portuguese")
    words = [re.sub(r"[^\w\s]", "", word) for word in words if word.isalpha()]
    words = [word for word in words if word not in stop_words]

    doc = nlp(" ".join(words))
    lemmatized_words = [token.lemma_ for token in doc]

    return " ".join(words)
