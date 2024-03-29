import json
import os,struct
from queue import Queue
from threading import Thread
from random import shuffle, sample
from tensorflow.core.example import example_pb2
import numpy as np


def sample_generator(bin_fname, single_pass=False):
    """
    Generator that reads binary file and yield text sample for training or inference.
    setname(str): binary file that one of ['train', 'dev', 'test']
    single_pass(boolean): If True, iterate the whole dataset only once (for testcase)
    """
    assert os.path.exists(bin_fname)

    while True:  # If single_pass, escape this loop after one epoch!
        reader = open(bin_fname, 'rb')
        while True:
            len_bytes = reader.read(8)
            if not len_bytes:
                #print("Reading one file is end!")
                break  # Break if file is end
            str_len = struct.unpack('q', len_bytes)[0]
            example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
            example = example_pb2.Example.FromString(example_str)

            if 'main' in bin_fname:
                w0 = example.features.feature['w0'].bytes_list.value[0].decode()
                w1 = example.features.feature['w1'].bytes_list.value[0].decode()
                claim = example.features.feature['claim'].bytes_list.value[0].decode()
                reason = example.features.feature['reason'].bytes_list.value[0].decode()
                try:
                    label = example.features.feature['label'].bytes_list.value[0].decode()
                except:
                    assert 'test' in bin_fname
                    label = '-1'
                yield (w0,w1,claim,reason,label)

            elif 'nli' in bin_fname:
                premise = example.features.feature['premise'].bytes_list.value[0].decode()
                hypothesis = example.features.feature['hypothesis'].bytes_list.value[0].decode()
                label = example.features.feature['label'].bytes_list.value[0].decode()
                yield (premise, hypothesis, label)
            else:
                raise ValueError
        if single_pass:
            print("Single pass is end!")
            break


class Example:
    def __init__(self, sent0, sent1, label, claim, reason, voca, hps):
        self.is_nli = True
        self.hps, self.vocab = hps, voca
        sent0, sent1 = sent0.split()[:self.hps.max_enc_len], sent1.split()[:self.hps.max_enc_len]
        self.sent0_len, self.sent1_len = len(sent0), len(sent1)
        self.claim_len, self.reason_len = 0, 0
        if label in '01':  # reasoning example
            label = int(label)
            self.is_nli = False
            claim, reason = claim.split()[:self.hps.max_enc_len], reason.split()[:self.hps.max_enc_len]
            self.claim_len, self.reason_len = len(claim), len(reason)
            self.claim_input = self.vocab.text2ids(claim)
            self.reason_input = self.vocab.text2ids(reason)
        else:
            assert claim is None and reason is None
            if label == 'neutral':
                label = 0
            elif label =='contradiction':
                label = 1
            elif label == 'entailment':
                label = 2
            else:
                raise ValueError

        self.label = label
        self.sent0_input = self.vocab.text2ids(sent0)
        self.sent1_input = self.vocab.text2ids(sent1)

        self.original_sent0_text = ' '.join(sent0)
        self.original_sent1_text = ' '.join(sent1)

    def pad_enc_input(self, max_len):
        while len(self.sent0_input) < max_len:
            self.sent0_input.append(self.vocab.pad_id)
        while len(self.sent1_input) < max_len:
            self.sent1_input.append(self.vocab.pad_id)
        if not self.is_nli:
            while len(self.claim_input) < max_len:
                self.claim_input.append(self.vocab.pad_id)
            while len(self.reason_input) < max_len:
                self.reason_input.append(self.vocab.pad_id)


class Batch():
    def __init__(self, example_list, hps, vocab):
        self.hps = hps
        self.vocab = vocab
        self.init_enc_seq(example_list)
        self.save_original_seq(example_list)

    def init_enc_seq(self, example_list):
        max_enc_len = self.hps.max_enc_len #max([ex.sent0_len for ex in example_list] + [ex.sent1_len for ex in example_list] + [ex.reason_len for ex in example_list] + [ex.claim_len for ex in example_list])
        for ex in example_list:
            ex.pad_enc_input(max_enc_len)

        label_num = 3 if example_list[0].is_nli else 2
        self.sent0_batch = np.zeros((self.hps.batch_size, max_enc_len), dtype=np.int32)
        self.sent0_lens = np.zeros((self.hps.batch_size), dtype=np.int32)
        self.sent1_batch = np.zeros((self.hps.batch_size, max_enc_len), dtype=np.int32)
        self.sent1_lens = np.zeros((self.hps.batch_size), dtype=np.int32)
        self.claim_batch = np.zeros((self.hps.batch_size, max_enc_len), dtype=np.int32)
        self.claim_lens = np.zeros((self.hps.batch_size), dtype=np.int32)
        self.reason_batch = np.zeros((self.hps.batch_size, max_enc_len), dtype=np.int32)
        self.reason_lens = np.zeros((self.hps.batch_size), dtype=np.int32)
        self.label = np.zeros((self.hps.batch_size,label_num), dtype=np.int32)

        # Fill enc batch
        for idx, ex in enumerate(example_list):
            self.sent0_batch[idx, :] = ex.sent0_input[:]
            self.sent0_lens[idx] = ex.sent0_len
            self.sent1_batch[idx, :] = ex.sent1_input[:]
            self.sent1_lens[idx] = ex.sent1_len
            self.label[idx][ex.label] = 1
            if not ex.is_nli:
                self.claim_batch[idx, :] = ex.claim_input[:]
                self.claim_lens[idx] = ex.claim_len
                self.reason_batch[idx, :] = ex.reason_input[:]
                self.reason_lens[idx] = ex.reason_len

    def save_original_seq(self, example_list):
        self.original_sent0_text = [ex.original_sent0_text for ex in example_list]
        self.original_sent1_text = [ex.original_sent1_text for ex in example_list]

class Batcher:
    def __init__(self, vocab, bin_path, hps):
        assert os.path.exists(bin_path)
        self.vocab = vocab
        self.bin_path = bin_path
        # bin_fname = args.split_data_path.format(setname).replace('.json', '.bin')
        self.hps = hps
        self.single_pass = True if 'eval' in hps.mode else False

        QUEUE_MAX_SIZE = 50
        self.batch_cache_size = 50
        self.batch_queue = Queue(QUEUE_MAX_SIZE)
        self.example_queue = Queue(QUEUE_MAX_SIZE * 16)#self.hps.batch_size)

        self.example_thread = Thread(target=self.fill_example_queue)
        self.example_thread.daemon = True
        self.example_thread.start()

        self.batch_thread = Thread(target=self.fill_batch_queue)
        self.batch_thread.daemon = True
        self.batch_thread.start()

    def next_batch(self):
        if self.batch_queue.qsize() == 0:
            if self.single_pass:
                print("[*]FINISH decoding")
                return 'FINISH'
            else:
                print("Batch queue is empty. waiting....")
                raise ValueError("Unexpected finish of batching.")
        batch = self.batch_queue.get()
        return batch

    def fill_example_queue(self):
        gen = sample_generator(self.bin_path, self.single_pass)
        while True:
            try:
                if 'nli' in self.bin_path:
                    premise, hypo, label = next(gen)
                    example = Example(premise, hypo, label, None, None, self.vocab, self.hps)
                elif 'main' in self.bin_path:
                    w0, w1, claim, reason, label = next(gen)
                    example = Example(w0, w1, label, claim, reason, self.vocab, self.hps)
                else:
                    raise ValueError

            except Exception as err:
                print("Error while fill example queue: {}".format(self.example_queue.qsize()))
                assert self.single_pass
                break
            self.example_queue.put(example)

    def fill_batch_queue(self):
        while True:
            if not self.single_pass:
                assert 'eval' not in self.hps.mode
                inputs = []
                for _ in range(self.hps.batch_size * self.batch_cache_size):
                    inputs.append(self.example_queue.get())

                batches = []
                for idx in range(0, len(inputs), self.hps.batch_size):
                    batches.append(inputs[idx:idx + self.hps.batch_size])
                if not self.single_pass:
                    shuffle(batches)
                for bat in batches:
                    self.batch_queue.put(Batch(bat, self.hps, self.vocab))
            else:
                assert 'eval' in self.hps.mode
                sample = self.example_queue.get()
                bat = [sample for _ in range(self.hps.batch_size)]
                self.batch_queue.put(Batch(bat, self.hps, self.vocab))
