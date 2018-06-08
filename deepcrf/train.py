# -*- coding=utf-8 -*-

import tensorflow as tf
import os
import shutil
from . import utils
from .model import BiLSTM_CRF
import logging


def copy_file(src, dst):
    fp = open(src, "r")
    buff = fp.read()
    fp.close()

    fp = open(dst, "w")
    fp.write(buff)
    fp.flush()
    fp.close()


def train(config, _transform_class, need_transform=False, rebuild_word2vec=False, restore_model=False):
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
          allow_soft_placement=config.allow_soft_placement,
          log_device_placement=config.log_device_placement)
        session_conf.gpu_options.allow_growth = True
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            transformer = _transform_class(config)
            if need_transform:
                transformer.transform(rebuild_word2vec)
            else:
                transformer.load()

            model = BiLSTM_CRF(sess, transformer, config, transformer.num_tags, transformer.vocab_size, transformer.word_vector)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(config.lr)
            grads_and_vars = optimizer.compute_gradients(model.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Keep track of gradient values and sparsity (optional)
            """
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)
            """

            saver = tf.train.Saver(max_to_keep=config.num_checkpoints)

            # Output directory for models and summaries
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs"))
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if restore_model:
                init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
                sess.run(init_op)
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                if os.path.exists(out_dir + "/checkpoints"):
                    shutil.rmtree(out_dir + "/checkpoints")
                if os.path.exists(out_dir + "/summaries"):
                    shutil.rmtree(out_dir + "/summaries")

            # Summaries for loss and acc
            loss_summary = tf.summary.scalar("loss", model.loss)
            if utils.tf_version_uper_than("1.3.0"):
                acc_summary = tf.summary.scalar("acc", model.accuracy)
                train_summary_op = tf.summary.merge([loss_summary, acc_summary]) # grad_summaries_merged
                dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            else:
                train_summary_op = tf.summary.merge([loss_summary]) # grad_summaries_merged
                dev_summary_op = tf.summary.merge([loss_summary])

            #Train Summaries
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)

            checkpoint_index_dev = os.path.join(out_dir, "summaries", "dev", "checkpoint")
            checkpoint_index = os.path.join(checkpoint_dir, "checkpoint")

            if not restore_model:
                init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
                sess.run(init_op)

            def train_step(doc_batch, labels_batch, seq_lens):
                feed_dict = {
                    model.input: doc_batch,
                    model.labels: labels_batch,
                    model.sequence_lengths: seq_lens,
                    model.input_keep_prob: config.input_keep_prob,
                    model.output_keep_prob: config.output_keep_prob,
                }

                if utils.tf_version_uper_than("1.3.0"):
                    _, step, summaries, loss, acc = sess.run([train_op, global_step, train_summary_op,
                                                              model.loss, model.accuracy], feed_dict)
                else:
                    _, step, summaries, loss, logits, transition_params = \
                        sess.run([train_op, global_step, train_summary_op,
                                  model.loss, model.logits,
                                  model.transition_params], feed_dict)
                    acc = model.acc(logits, transition_params, seq_lens, labels_batch)

                logging.info("step {}, loss {:g}, acc {:g}".format(step, loss, acc))
                train_summary_writer.add_summary(summaries, step)


            def dev_step(doc_batch, labels_batch, seq_lens, writer=None):
                feed_dict = {
                    model.input: doc_batch,
                    model.labels: labels_batch,
                    model.sequence_lengths: seq_lens,
                    model.input_keep_prob: 1.,
                    model.output_keep_prob: 1.,
                }

                if utils.tf_version_uper_than("1.3.0"):
                    step, summaries, loss, acc = sess.run([global_step, dev_summary_op, model.loss,
                                                           model.accuracy], feed_dict)
                else:
                    step, summaries, loss, logits, transition_params = \
                        sess.run([global_step, dev_summary_op, model.loss,
                                  model.logits, model.transition_params], feed_dict)
                    acc = model.acc(logits, transition_params, seq_lens, labels_batch)

                logging.info("step {}, loss {:g}, acc {:g}".format(step, loss, acc))
                if writer:
                    writer.add_summary(summaries, step)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord, sess=sess)
            try:
                while not coord.should_stop():
                    labels_batch, doc_batch, seq_lens = transformer.pull_batch(sess)
                    train_step(doc_batch, labels_batch, seq_lens)
                    current_step = tf.train.global_step(sess, global_step)
                    if current_step % config.evaluate_every == 0:
                        labels_batch, doc_batch, seq_lens = transformer.get_test_data()
                        logging.info("\nEvaluation:")
                        dev_step(doc_batch, labels_batch, seq_lens, writer=dev_summary_writer)
                        logging.info("")
                    if current_step % config.checkpoint_every == 0:
                        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                        copy_file(checkpoint_index, checkpoint_index_dev)
                        logging.info("Saved model checkpoint to {}\n".format(path))
            except tf.errors.OutOfRangeError:
                logging.info ('Done training -- epoch limit reached')

            coord.request_stop()
            coord.join(threads)
            return model

