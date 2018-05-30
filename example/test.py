
from deepcrf import DefaultCorpusIterator

corpus = DefaultCorpusIterator(["data/test.txt"])

for parts in corpus:
    print(parts)