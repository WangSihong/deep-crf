from .transform import DeepCRFTransform
from .utils import CorpusIterator

class DefaultCorpusIterator(CorpusIterator):
    def __init__(self, files):
        super(DefaultCorpusIterator, self).__init__(files)
        self.parts = []

    def analysis_line(self, line):
        line = line.strip()
        if line == "":
            parts = self.parts
            self.parts = []
            return parts
        
        self.parts.append(line.split("\t")[0])


class DefaultTransform(DeepCRFTransform):
    def __init__(self, config):
        super(DefaultTransform, self).__init__(config=config,
                                               corpus_iter=DefaultCorpusIterator([config.train_input_path]))
        self.parts = []
        self.tags = []

    def tag_line(self, line):
        line = line.strip()
        if line == "":
            parts = self.parts
            tags = self.tags
            self.parts = []
            self.tags = []
            return parts, tags
        
        ps = line.split("\t")
        if len(ps) >= 2:
            self.parts.append(ps[0])
            self.tags.append(ps[1])
        return [], []
