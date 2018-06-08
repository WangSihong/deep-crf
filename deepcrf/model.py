import tensorflow as tf
from . import utils
import numpy as np
import logging


def _load_graph(model_path):
    with tf.gfile.GFile(model_path, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name = "")
    return graph


class DeepCRFModel(object):
    def __init__(self, session, transformer):
        self.session = session
        self.transformer = transformer
        self.input = None
        self.sequence_lengths = None
        self.input_keep_prob = None
        self.output_keep_prob = None
        if utils.tf_version_uper_than("1.3.0"):
            self.viterbi_sequence = None
            self.viterbi_score = None
        else:
            self.logits = None
            self.transition_params = None

    def predict(self, multi_parts):
        part_ids = self.transformer.transform_parts(multi_parts)
        part_lens = [len(parts) for parts in multi_parts]
        feed_dict = {self.input: part_ids,
                     self.sequence_lengths: np.array(part_lens),
                     self.input_keep_prob: 1.0,
                     self.output_keep_prob: 1.0}

        if utils.tf_version_uper_than("1.3.0"):
            viterbi_sequences, viterbi_scores = self.session.run([self.viterbi_sequence, self.viterbi_score], feed_dict=feed_dict)
            tags = self.transformer.transform_tags(viterbi_sequences)
        else:
            logits, trans = self.session.run([self.logits, self.transition_params], feed_dict=feed_dict)
            viterbi_sequences = self.decode(logits, trans, part_lens)
            tags = self.transformer.transform_tags(viterbi_sequences)

        labels = []
        for parts, seqs in zip(multi_parts, tags):
            ret = []
            for part, tag in zip(parts, seqs):
                ret.append([part, tag])
            labels.append(ret)

        return labels

    def decode(self, logits, transition_params, sequence_lengths):
        logits_list = np.split(logits, logits.shape[0])
        viterbi_sequences = []

        for score, seq_len in zip(logits_list, sequence_lengths):
            viterbi_sequence, viterbi_score = tf.contrib.crf.viterbi_decode(np.reshape(score, [score.shape[1], score.shape[2]]), transition_params)
            viterbi_sequences.append(viterbi_sequence[:seq_len])

        return viterbi_sequences
    
    def eval(self, line_iter):
        rets = []
        total = 0
        correct = 0
        for line in line_iter:
            parts, tags = self.transformer.tag_line(line)
            if len(parts) == 0:
                continue
            mtags = [tags]
            pred_ret = self.predict([parts])
            rets.append(pred_ret)
            for tags, pred in zip(mtags, pred_ret):
                ptags = [p[1] for p in pred]
                for to, tp in zip(tags, ptags):
                    total += 1
                    if to == tp:
                        correct += 1
        logging.info("[Eval] total: %d , correct: %d, accuracy: %f" % (total, correct, float(correct)/float(total)) )
        return rets
        

class GraphDeepCRFModel(DeepCRFModel):
    def __init__(self, model_path, config, TransformClass):
        super(GraphDeepCRFModel, self).__init__(tf.Session(graph=_load_graph(model_path)), TransformClass(config))
        self.transformer.load_for_predict()
        self.session.as_default()
        self.input = self.session.graph.get_tensor_by_name("input:0")
        self.sequence_lengths = self.session.graph.get_tensor_by_name("sequence_lengths:0")
        self.input_keep_prob = self.session.graph.get_tensor_by_name("input_keep_prob:0")
        self.output_keep_prob = self.session.graph.get_tensor_by_name("output_keep_prob:0")
        if utils.tf_version_uper_than("1.3.0"):
            self.viterbi_sequence = self.session.graph.get_tensor_by_name("viterbi/cond/Merge:0")
            self.viterbi_score = self.session.graph.get_tensor_by_name("viterbi/cond/Merge_1:0")
        else:
            self.logits = self.session.graph.get_tensor_by_name("logits/logits:0")
            self.transition_params = self.session.graph.get_tensor_by_name("crf/transitions:0")


class BiLSTM_CRF(DeepCRFModel):
    def __init__(self, session, transformer, config, num_tags, vocab_size, embedding_data):
        super(BiLSTM_CRF, self).__init__(session, transformer)
        self.config = config
        self.input = tf.placeholder(tf.int32, shape=[None, None], name="input")
        self.labels = tf.placeholder(tf.int32, shape=[None, None], name="labels")
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None], name="sequence_lengths")
        self.input_keep_prob = tf.placeholder(tf.float32, name="input_keep_prob")
        self.output_keep_prob = tf.placeholder(tf.float32, name="output_keep_prob")

        def rnn_cell():
            rnn = tf.nn.rnn_cell.BasicRNNCell(num_units=config.hidden_size)
            rnn = tf.contrib.rnn.DropoutWrapper(cell=rnn, input_keep_prob=self.input_keep_prob,
                                                output_keep_prob=self.output_keep_prob)
            return rnn

        def lstm_cell():
            lstm = tf.contrib.rnn.BasicLSTMCell(config.hidden_size, forget_bias=1.0,
                                                state_is_tuple=True)  # reuse=tf.get_variable_scope().reuse
            lstm = tf.contrib.rnn.DropoutWrapper(cell=lstm, input_keep_prob=self.input_keep_prob,
                                                 output_keep_prob=self.output_keep_prob)
            return lstm

        def gru_cell():
            gru = tf.contrib.rnn.GRUCell(config.hidden_size)
            gru = tf.contrib.rnn.DropoutWrapper(cell=gru, input_keep_prob=self.input_keep_prob,
                                                output_keep_prob=self.output_keep_prob)
            return gru

        # embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            embedding_initializer = tf.constant_initializer(embedding_data)
            self._W = tf.get_variable("W", shape=[vocab_size, config.embedding_size],
                                      initializer=embedding_initializer,
                                      dtype=tf.float32)
            inputs = tf.nn.embedding_lookup(self._W, self.input)

        # fw_cell = tf.contrib.rnn.MultiRNNCell([gru_cell() for _ in range(self.hidden_layer_num)], state_is_tuple=True)
        # bw_cell = tf.contrib.rnn.MultiRNNCell([gru_cell() for _ in range(self.hidden_layer_num)], state_is_tuple=True)
        fw_cell = gru_cell()
        bw_cell = gru_cell()

        (output_fw_seq, output_bw_seq), _ = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, inputs,
                                                                            sequence_length=self.sequence_lengths,
                                                                            dtype=tf.float32,
                                                                            scope='dynamic_rnn_layer')
        output = tf.concat([output_fw_seq, output_bw_seq], axis=-1, name="output")

        with tf.variable_scope("logits"):
            W = tf.get_variable("W", shape=[2 * config.hidden_size, num_tags],
                                initializer=tf.contrib.layers.xavier_initializer(),
                                dtype=tf.float32)

            b = tf.get_variable("b", shape=[num_tags],
                                initializer=tf.zeros_initializer(),
                                dtype=tf.float32)

            s = tf.shape(output)
            output = tf.reshape(output, [-1, 2*config.hidden_size])
            prob = tf.nn.xw_plus_b(output, W, b, name="prob")
            self.logits = tf.reshape(prob, [-1, s[1], num_tags], name="logits")

        with tf.variable_scope("crf"):
            log_likelihood, transition_params = \
                tf.contrib.crf.crf_log_likelihood(inputs=self.logits,
                                                  tag_indices=self.labels,
                                                  sequence_lengths=self.sequence_lengths)

        if utils.tf_version_uper_than("1.3.0"):
            with tf.variable_scope("viterbi"):
                viterbi_sequence, viterbi_score = \
                    tf.contrib.crf.crf_decode(self.logits, transition_params, self.sequence_lengths)
                self.viterbi_sequence = viterbi_sequence
                self.viterbi_score = viterbi_score
            with tf.name_scope("accuracy"):
                mask = tf.sequence_mask(self.sequence_lengths)
                prediction = tf.boolean_mask(self.viterbi_sequence, mask)
                target = tf.boolean_mask(self.labels, mask)
                correct_prediction = tf.equal(prediction, target)
                self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")
        else:
            self.transition_params = transition_params
            
        with tf.name_scope("loss"):
            self.loss = -tf.reduce_mean(log_likelihood, name="loss")

    def acc(self, logits, transition_params, sequence_lengths, target_labels):
        logits_list = np.split(logits, logits.shape[0])
        viterbi_sequences = []

        for score in logits_list:
            viterbi_sequence, viterbi_score = tf.contrib.crf.viterbi_decode(np.reshape(score, [score.shape[1], score.shape[2]]), transition_params)
            viterbi_sequences.append(viterbi_sequence)

        total = np.sum(sequence_lengths)
        correct = 0
        for predict, target, seq_len in zip(viterbi_sequences, target_labels, sequence_lengths):
            for i in range(seq_len):
                if predict[i] == target[i]:
                    correct += 1
        return float(correct)/float(total)

    def save(self, output_path):
        save_graph_value_scopes = []
        if utils.tf_version_uper_than("1.3.0"):
            save_graph_value_scopes.append(self.viterbi_sequence.name.split(":")[0])
            save_graph_value_scopes.append(self.viterbi_score.name.split(":")[0])
        else:
            save_graph_value_scopes.append(self.transition_params.name.split(":")[0])
         
        save_graph_value_scopes.append("logits/prob")
        save_graph_value_scopes.append("logits/logits")
        logging.info(save_graph_value_scopes)
        graph = tf.graph_util.convert_variables_to_constants(self.session, self.session.graph_def, save_graph_value_scopes)
        with tf.gfile.GFile(output_path, "wb") as f:  
            f.write(graph.SerializeToString())

