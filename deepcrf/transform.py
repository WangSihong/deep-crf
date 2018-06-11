# -*- coding=utf-8 -*-
import os
import shutil
from abc import abstractmethod

import numpy as np
import tensorflow as tf
from gensim.models import word2vec
from gensim.models import Word2Vec
import json
from .utils import CorpusIterator
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class Transform(object):
    def __init__(self, config, temp_dir, corpus_iter):
        self.config = config
        self.train_input_path = config.train_input_path
        self.test_input_path = config.test_input_path
        self.temp_path = temp_dir
        self.data_path = self.temp_path + "/data"

        self.tfrecord_path = self.data_path + "/tfrecord.bin"

        # vocab
        self.id2w = {}
        self.w2id = {}
        self.vocab_path = self.data_path + "/vocab.txt"
        self.vocab_size = 0

        # word2vec
        self.word_vector = np.array([])
        self.word_vector_path = self.temp_path + "/word2vec/word_vector.npy"
        self.w2v_model_path = self.temp_path + "/word2vec/model.bin"
        self.corpus_iter = corpus_iter

    def build_word2vec(self):
        logging.info("Build word2vec...")
        model = Word2Vec(self.corpus_iter,
                         size=self.config.embedding_size,
                         alpha=self.config.alpha,
                         window=self.config.window,
                         min_count=self.config.min_count,
                         sample=self.config.sample,
                         workers=self.config.workers,
                         negative=self.config.negative,
                         iter=self.config.iter_times,
                         seed=3)

        model.save(self.w2v_model_path)

    def get_word2vec(self, w2v_model, word):
        w2v_model.init_sims()
        if word in w2v_model.wv.vocab:
            return w2v_model.wv.word_vec(word, use_norm=True)
        return np.array([0.0] * w2v_model.layer1_size)

    def sparse_to_dense(self,
                        sparse_indices,
                        output_shape,
                        sparse_values,
                        default_value=0):
        """
        sparse_indices: A 0-D, 1-D, or 2-D `Tensor` of type `int32` or `int64`.
          `sparse_indices[i]` contains the complete index where `sparse_values[i]`
          will be placed.
        output_shape: A 1-D `Tensor` of the same type as `sparse_indices`.  Shape
          of the dense output tensor.
        sparse_values: A 0-D or 1-D `Tensor`.  Values corresponding to each row of
          `sparse_indices`, or a scalar value to be used for all sparse indices.
        default_value: A 0-D `Tensor` of the same type as `sparse_values`.  Value
          to set for indices not specified in `sparse_indices`.  Defaults to zero.
        """

        arr = np.full(output_shape, default_value, dtype=np.int64)
        size = len(sparse_indices)
        for i in range(size):
            arr[sparse_indices[i][0]][sparse_indices[i][1]] = sparse_values[i]
        return arr

    def nparray_index(self, arr, ele=-1):
        for i in range(len(arr)):
            if arr[i] == ele:
                return i
        return len(arr) + 1

    def load_vocab(self):
        if not os.path.exists(self.vocab_path):
            return

        count = 1
        with open(self.vocab_path, 'r') as fp:
            for line in fp:
                line = line.strip()
                self.id2w[count] = line
                self.w2id[line] = count
                count += 1
        self.vocab_size = count
        logging.info("Load vocab, size %d" % (self.vocab_size))

    def pad(self, ndarr, pad_len, value=0):
        arr = ndarr[:self.nparray_index(ndarr)]
        fill_size = pad_len - len(arr)
        if fill_size > 0:
            pad_fill = [value] * fill_size
            return np.append(arr, pad_fill)
        elif fill_size < 0:
            return arr[:pad_len]
        else:
            return arr

    def clean_data(self):
        if os.path.exists(self.data_path):
            shutil.rmtree(self.data_path)
        os.mkdir(self.data_path)

    def load_pre_data(self):
        self.load_vocab()
        self.load_custom_data()

    def transform(self, rebuild_word2vec=False):
        logging.info("Transform data...")
        self.load_pre_data()

        if self.corpus_iter is not None and (rebuild_word2vec or not os.path.exists(self.w2v_model_path)):
            if os.path.exists(self.temp_path + "/word2vec"):
                shutil.rmtree(self.temp_path + "/word2vec")
            os.mkdir(self.temp_path + "/word2vec")
            self.build_word2vec()

        self.transform_impl()

    def load(self):
        self.load_pre_data()

        if os.path.exists(self.word_vector_path):
            self.word_vector = np.load(self.word_vector_path)

        if os.path.exists(self.tfrecord_path):
            self.load_tfrecords_data()

        if os.path.exists(self.test_input_path):
            self.load_test_data()

    @abstractmethod
    def transform_impl(self):
        pass

    @abstractmethod
    def load_test_data(self):
        pass

    @abstractmethod
    def load_tfrecords_data(self):
        pass

    @abstractmethod
    def load_custom_data(self):
        pass

    @abstractmethod
    def pull_batch(self, sess):
        pass

    @abstractmethod
    def get_test_data(self):
        pass


class DeepCRFTransform(Transform):
    def __init__(self, config, temp_dir="runs", corpus_iter=None):
        super(DeepCRFTransform, self).__init__(config, temp_dir, corpus_iter)

        self.tag_path = self.data_path + "/tags.txt"

        self.tag2id = {}
        self.id2tag = {}
        self.tag_queue = None
        self.text_queue = None
        self.seq_lens_queue = None
        self.test_tag_data = np.array([])
        self.test_text_data = np.array([])
        self.test_lengths = np.array([])
        self.num_tags = 0

    def load_custom_data(self):
        if not os.path.exists(self.tag_path):
            return
        
        count = 0
        with open(self.tag_path, 'r') as fp:
            for line in fp:
                line = line.strip()
                self.id2tag[count] = line
                self.tag2id[line] = count
                count += 1
        self.num_tags = count
        logging.info("Load tags data, size %d" % (self.num_tags))

    def load_test_data(self):
        with open(self.test_input_path, "r", errors="ignore") as fp:
            logging.info("Load test data %s" % (self.test_input_path))
            wss = []
            tss = []
            max_len = 0
            for line in fp:
                ws, ts = self.tag_line(line)
                if ws is None or len(ws) == 0:
                    continue
                if self.config.seq_cut_length > 0 and self.config.seq_cut_length < len(ws):
                    ws = ws[:self.config.seq_cut_length]
                    ts = ts[:self.config.seq_cut_length]
                wss.append(ws)
                tss.append(ts)
                if len(ws) > max_len:
                    max_len = len(ws)
                        
            test_data_tags = []
            test_data_text = []
            test_data_leng = []
            for ws, ts in zip(wss, tss):
                tags = [self.tag2id[t] for t in ts]
                words = [self.w2id.get(w, 0) for w in ws]
                test_data_leng.append(len(tags))

                tags = self.pad(tags, max_len, 0)
                words = self.pad(words, max_len, 0)
                test_data_tags.append(tags)
                test_data_text.append(words)
            self.test_tag_data = np.array(test_data_tags)
            self.test_text_data = np.array(test_data_text)
            self.test_lengths = np.array(test_data_leng)
            logging.info("Load test data, size %d" % (self.test_tag_data.shape[0]))
         
    def load_tfrecords_data(self):
        logging.info("Load tfrecord data %s", self.tfrecord_path)
        filename_queue = tf.train.string_input_producer([self.tfrecord_path], num_epochs=self.config.epoch)
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(
            serialized_example,
            features={
                'tag': tf.VarLenFeature(tf.int64),
                'text': tf.VarLenFeature(tf.int64),
                'seq_lens': tf.FixedLenFeature([], tf.int64)
            })

        tag = features['tag']
        text = features['text']
        seq_lens = features['seq_lens']

        tag_q, text_q, seq_lens_q = tf.train.shuffle_batch([tag, text, seq_lens], batch_size=self.config.batch_size, capacity=1000,
                                                            num_threads=4, min_after_dequeue=500)
        self.tag_queue = tag_q
        self.text_queue = text_q
        self.seq_lens_queue = seq_lens_q

    def transform_parts(self, multi_parts):
        mps = []
        max_len = max(map(lambda x: len(x), multi_parts))
        for parts in multi_parts:
            ids = np.array([self.w2id.get(p, 0) for p in parts])
            mps.append(self.pad(ids, max_len, 0))
        return np.array(mps)

    def transform_tags(self, multi_labels):
        tags = []
        for labels in multi_labels:
            tags.append([self.id2tag.get(l, 0) for l in labels])
        return tags

    def pull_batch(self, sess):
        _tag_sparse, _text_sparse, _seq_lens = sess.run([self.tag_queue, self.text_queue, self.seq_lens_queue])
        _tags = self.sparse_to_dense(_tag_sparse.indices, _tag_sparse.dense_shape, _tag_sparse.values, -1)
        _texts = self.sparse_to_dense(_text_sparse.indices, _text_sparse.dense_shape, _text_sparse.values, -1)

        if self.config.seq_cut_length > 0:
            _seq_lens[np.where(_seq_lens > self.config.seq_cut_length)] = self.config.seq_cut_length

        max_len = max(_seq_lens)

        tags = []
        texts = []
        for i in range(self.config.batch_size):
            texts.append(self.pad(_texts[i], max_len, 0))
            tags.append(self.pad(_tags[i], max_len, 0))

        return np.array(tags), np.array(texts), _seq_lens
    
    def get_test_data(self):
        return self.test_tag_data, self.test_text_data, self.test_lengths

    def transform_impl(self):
        logging.info("Build tfrecord file...")
        
        if not os.path.exists(self.data_path):
            os.mkdir(self.data_path)
            
        writer = tf.python_io.TFRecordWriter(self.tfrecord_path)
        word_count = 1  # 0 is UNK
        tag_count = 0

        vocab_fp = open(self.vocab_path, "a")
        tag_fp = open(self.tag_path, "a")

        w2v_model = word2vec.Word2Vec.load(self.w2v_model_path)
        w2v_layer_size = w2v_model.layer1_size

        counter = 0
        vecs = [0.0] * w2v_layer_size
        with open(self.train_input_path, "r", errors="ignore") as fp:
            for line in fp:
                line = line.strip()
                ws, ts = self.tag_line(line)

                counter += 1
                if counter % 10000 == 0:
                    logging.info("Transform %d" % (counter))
                    
                if ws is None or len(ws) == 0:
                    continue

                for w in ws:
                    if w not in self.w2id:
                        self.w2id[w] = word_count
                        self.id2w[word_count] = w

                        v = self.get_word2vec(w2v_model, w)
                        vecs.extend(v)

                        word_count += 1
                        vocab_fp.write(w + "\n")

                for t in ts:
                    if t not in self.tag2id:
                        self.tag2id[t] = tag_count
                        self.id2tag[tag_count] = t
                        tag_count += 1
                        tag_fp.write(t + "\n")

                tags = [self.tag2id[t] for t in ts]
                words = [self.w2id[w] for w in ws]
                example = tf.train.Example(features=tf.train.Features(feature={
                    'tag': tf.train.Feature(int64_list=tf.train.Int64List(value=tags)),
                    'text': tf.train.Feature(int64_list=tf.train.Int64List(value=words)),
                    'seq_lens': tf.train.Feature(int64_list=tf.train.Int64List(value=[len(words)]))
                }))
                writer.write(example.SerializeToString())

        self.vocab_size = word_count
        self.num_tags = tag_count
        self.word_vector = np.array(vecs).reshape([word_count, w2v_layer_size])
        np.save(self.word_vector_path, self.word_vector)

        vocab_fp.close()
        tag_fp.close()
        writer.close()

        if os.path.exists(self.tfrecord_path):
            self.load_tfrecords_data()
        
        if os.path.exists(self.test_input_path):
            self.load_test_data()
            
    @abstractmethod
    def tag_line(self, line):
        pass

