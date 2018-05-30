# -*- coding:utf-8 -*-

import tensorflow as tf
from abc import abstractmethod


class CorpusIterator(object):
    def __init__(self, files):
        self.files = files

    def __iter__(self):
        for fn in self.files:
            with open(fn, "r") as fp:
                for line in fp:
                    rets = self.analysis_line(line) 
                    if rets is not None and len(rets) > 0:
                        yield rets

    @abstractmethod
    def analysis_line(self, line):
        pass


def tf_version_uper_than(version):
    tf_vps = [p for p in tf.__version__.split(".")]
    vps = [p for p in version.split(".")]
    tf_vp = [int(tf_vps[0]), int(tf_vps[1])]
    vp = [int(vps[0]), int(vps[1])]
    if tf_vp[0] > vp[0]:
        return True
    elif tf_vp[0] == vp[0] and tf_vp[1] > vp[1]:
        return True
    return False
