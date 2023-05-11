import spacy

nlp = spacy.load("en_core_web_sm")

text = "This is a sentence. This is another sentence."

doc = nlp(text)

for sent in doc.sents:
    print(sent.text)