
from deepcrf import DeepCRFTransform
from deepcrf import CorpusIterator

class CustomCorpusIterator(CorpusIterator):
    def __init__(self, files):
        super(OceanusSegmentCorpusIterator, self).__init__(files)

    def analysis_line(self, line):
        chars = []
        """
          line analysis code here.
        """
        return chars


class CustomTransform(DeepCRFTransform):
    def __init__(self, config):
        super(OceanusSegmentTransform, self).__init__(config=config,
                                                      corpus_iter=CustomCorpusIterator([config.train_input_path]))

    def tag_line(self, line):
        chars = []
        tags = []
        """
          line analysis code here.
        """
        return chars, tags
