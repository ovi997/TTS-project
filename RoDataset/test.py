from phonemizer.backend import EspeakBackend
from phonemizer.punctuation import Punctuation
from phonemizer.separator import Separator

text = "Când am început să strig, cerul s-a luminat. Era o toamnă târzie, dar mai vedeai câteva fire de flori pe-alocuri. Numărul 14 este câștigător?"
# remove all the punctuation from the text, considering only the specified
# punctuation mark
# text = Punctuation(';:,.!"?()-').remove(text)

print(text)

# build the set of all the words in the text
words = {w.lower() for line in text for w in line.strip().split(' ') if w}

print(words)
# initialize the espeak backend for English
backend = EspeakBackend('ro')

# separate phones by a space and ignoring words boundaries
separator = Separator(phone=' ', word=None)

# build the lexicon by phonemizing each word one by one. The backend.phonemize
# function expect a list as input and outputs a list.
lexicon = backend.phonemize([text], separator=separator, strip=True)

print(lexicon)