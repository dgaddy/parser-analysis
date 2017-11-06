import functools
import random

import dynet as dy
import numpy as np

import trees

START = "<START>"
STOP = "<STOP>"
UNK = "<UNK>"

def augment(scores, oracle_index):
    assert isinstance(scores, dy.Expression)
    shape = scores.dim()[0]
    assert len(shape) == 1
    increment = np.ones(shape)
    increment[oracle_index] = 0
    return scores + dy.inputVector(increment)

def shuffle(items, start, end):
    if end <= start:
        return
    to_shuffle = items[start:end]
    random.shuffle(to_shuffle)
    items[start:end] = to_shuffle

def transpose_lists(nested_list):
    result = []
    for i in range(len(nested_list[0])):
        result.append([l[i] for l in nested_list])
    return result

class Feedforward(object):
    def __init__(self, model, input_dim, hidden_dims, output_dim):
        self.spec = locals()
        self.spec.pop("self")
        self.spec.pop("model")

        self.model = model.add_subcollection("Feedforward")

        self.weights = []
        self.biases = []
        dims = [input_dim] + hidden_dims + [output_dim]
        for prev_dim, next_dim in zip(dims, dims[1:]):
            self.weights.append(self.model.add_parameters((next_dim, prev_dim)))
            self.biases.append(self.model.add_parameters(next_dim))

    def param_collection(self):
        return self.model

    @classmethod
    def from_spec(cls, spec, model):
        return cls(model, **spec)

    def __call__(self, x):
        for i, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            weight = dy.parameter(weight)
            bias = dy.parameter(bias)
            x = dy.affine_transform([bias, weight, x])
            if i < len(self.weights) - 1:
                x = dy.rectify(x)
        return x

class TopDownParser(object):
    def __init__(
            self,
            model,
            tag_vocab,
            word_vocab,
            label_vocab,
            tag_embedding_dim,
            word_embedding_dim,
            lstm_layers,
            lstm_dim,
            label_hidden_dim,
            split_hidden_dim,
            dropout,
            lstm_type,
            lstm_context_size,
    ):
        self.spec = locals()
        self.spec.pop("self")
        self.spec.pop("model")

        self.model = model.add_subcollection("Parser")
        self.tag_vocab = tag_vocab
        self.word_vocab = word_vocab
        self.label_vocab = label_vocab
        self.lstm_dim = lstm_dim

        self.tag_embeddings = self.model.add_lookup_parameters(
            (tag_vocab.size, tag_embedding_dim), name="tag-embeddings")
        self.word_embeddings = self.model.add_lookup_parameters(
            (word_vocab.size, word_embedding_dim), name="word-embeddings")

        self.lstm = dy.BiRNNBuilder(
            lstm_layers,
            tag_embedding_dim + word_embedding_dim,
            2 * lstm_dim,
            self.model,
            dy.VanillaLSTMBuilder)

        self.f_label = Feedforward(
            self.model, 2 * lstm_dim, [label_hidden_dim], label_vocab.size)
        self.f_split = Feedforward(
            self.model, 2 * lstm_dim, [split_hidden_dim], 1)

        self.dropout = dropout

        self.lstm_type = lstm_type
        self.lstm_context_size = lstm_context_size

    def param_collection(self):
        return self.model

    @classmethod
    def from_spec(cls, spec, model):
        return cls(model, **spec)

    def get_basic_span_encoding(self, embeddings):
        lstm_outputs = self.lstm.transduce(embeddings)

        @functools.lru_cache(maxsize=None)
        def span_encoding(left, right):
            forward = (
                lstm_outputs[right][:self.lstm_dim] -
                lstm_outputs[left][:self.lstm_dim])
            backward = (
                lstm_outputs[left + 1][self.lstm_dim:] -
                lstm_outputs[right + 1][self.lstm_dim:])
            return dy.concatenate([forward, backward])

        return span_encoding

    def get_truncated_span_encoding(self, embeddings, distance):
        padded_embeddings = [embeddings[0]]*(distance-1)+embeddings+[embeddings[-1]]*(distance-1)
        forward_reps = []
        backward_reps = []
        for i in range(len(embeddings)-1):
            lstm_inputs = padded_embeddings[i:i+distance*2]
            lstm_outputs = self.lstm.transduce(lstm_inputs)
            forward_reps.append(lstm_outputs[distance-1][:self.lstm_dim])
            backward_reps.append(lstm_outputs[distance][self.lstm_dim:])

        @functools.lru_cache(maxsize=None)
        def span_encoding(left, right):
            forward = (
                forward_reps[right] -
                forward_reps[left])
            backward = (
                backward_reps[left] -
                backward_reps[right])
            return dy.concatenate([forward, backward])

        return span_encoding
    def get_truncated_span_encoding_batch(self, embeddings, distance):
        padded_embeddings = [embeddings[0]]*(distance-1)+embeddings+[embeddings[-1]]*(distance-1)
        batched_embeddings = []
        for i in range(distance*2):
            selected = padded_embeddings[i:len(padded_embeddings)-(distance*2)+i+1]
            catted = dy.concatenate_to_batch(selected)
            batched_embeddings.append(catted)
        assert batched_embeddings[0].dim()[1] == len(embeddings)-1 # batch dimension is length of sentence + 1

        lstm_outputs = self.lstm.transduce(batched_embeddings)

        forward_reps = lstm_outputs[distance-1][:self.lstm_dim]
        backward_reps = lstm_outputs[distance][self.lstm_dim:]

        @functools.lru_cache(maxsize=None)
        def span_encoding(left, right):
            forward = (
                dy.pick_batch_elem(forward_reps, right) -
                dy.pick_batch_elem(forward_reps, left))
            backward = (
                dy.pick_batch_elem(backward_reps, left) -
                dy.pick_batch_elem(backward_reps, right))
            return dy.concatenate([forward, backward])

        return span_encoding

    def get_shuffled_span_encoding(self, embeddings, distance):
        all_spans = []
        all_lstm_outputs = []
        for i in range(len(embeddings)-2):
            for j in range(i+1,len(embeddings)-1):
                all_spans.append((i,j))
                lstm_inputs = embeddings[:] # copy
                shuffle(lstm_inputs, 1, i-distance+1) # note not shuffling start/end padding
                shuffle(lstm_inputs, i+distance+1, j-distance+1)
                shuffle(lstm_inputs, j+distance+1, len(embeddings)-1)
                all_lstm_outputs.append(self.lstm.transduce(lstm_inputs))
        span_map = {span:idx for idx, span in enumerate(all_spans)}

        @functools.lru_cache(maxsize=None)
        def span_encoding(left, right):
            lstm_outputs = all_lstm_outputs[span_map[(left,right)]]
            forward = (
                lstm_outputs[right][:self.lstm_dim] -
                lstm_outputs[left][:self.lstm_dim])
            backward = (
                lstm_outputs[left + 1][self.lstm_dim:] -
                lstm_outputs[right + 1][self.lstm_dim:])
            return dy.concatenate([forward, backward])

        return span_encoding

    def get_shuffled_span_encoding_batch(self, embeddings, distance):
        all_spans = []
        all_lstm_inputs = []
        for i in range(len(embeddings)-2):
            for j in range(i+1,len(embeddings)-1):
                all_spans.append((i,j))
                lstm_inputs = embeddings[:] # copy
                shuffle(lstm_inputs, 1, i-distance+1) # note not shuffling start/end padding
                shuffle(lstm_inputs, i+distance+1, j-distance+1)
                shuffle(lstm_inputs, j+distance+1, len(embeddings)-1)
                all_lstm_inputs.append(lstm_inputs)
        span_map = {span:idx for idx, span in enumerate(all_spans)}

        all_lstm_inputs = [dy.concatenate_to_batch(items) for items in transpose_lists(all_lstm_inputs)]
        all_lstm_outputs = self.lstm.transduce(all_lstm_inputs)

        @functools.lru_cache(maxsize=None)
        def span_encoding(left, right):
            batch_index = span_map[(left,right)]
            forward = (
                dy.pick_batch_elem(all_lstm_outputs[right], batch_index)[:self.lstm_dim] -
                dy.pick_batch_elem(all_lstm_outputs[left], batch_index)[:self.lstm_dim])
            backward = (
                dy.pick_batch_elem(all_lstm_outputs[left + 1], batch_index)[self.lstm_dim:] -
                dy.pick_batch_elem(all_lstm_outputs[right + 1], batch_index)[self.lstm_dim:])
            return dy.concatenate([forward, backward])

        return span_encoding

    def get_inside_span_encoding(self, embeddings, distance, shuffle_inside=False):
        padded_embeddings = [embeddings[0]]*distance+embeddings+[embeddings[-1]]*distance
        all_spans = []
        all_lstm_outputs = []
        for i in range(len(embeddings)-2):
            for j in range(i+1,len(embeddings)-1):
                all_spans.append((i,j))
                lstm_inputs = padded_embeddings[i+1:j+1+2*distance]
                if shuffle_inside:
                    shuffle(lstm_inputs, 2*distance, len(lstm_inputs)-2*distance)
                all_lstm_outputs.append(self.lstm.transduce(lstm_inputs))
        span_map = {span:idx for idx, span in enumerate(all_spans)}

        @functools.lru_cache(maxsize=None)
        def span_encoding(left, right):
            lstm_outputs = all_lstm_outputs[span_map[(left,right)]]
            forward = (
                lstm_outputs[-distance-1][:self.lstm_dim] -
                lstm_outputs[distance-1][:self.lstm_dim])
            backward = (
                lstm_outputs[distance][self.lstm_dim:] -
                lstm_outputs[-distance][self.lstm_dim:])
            return dy.concatenate([forward, backward])

        return span_encoding

    def parse(self, sentence, gold=None, explore=True):
        is_train = gold is not None

        if is_train:
            self.lstm.set_dropout(self.dropout)
        else:
            self.lstm.disable_dropout()

        embeddings = []
        for tag, word in [(START, START)] + sentence + [(STOP, STOP)]:
            tag_embedding = self.tag_embeddings[self.tag_vocab.index(tag)]
            if word not in (START, STOP):
                count = self.word_vocab.count(word)
                if not count or (is_train and np.random.rand() < 1 / (1 + count)):
                    word = UNK
            word_embedding = self.word_embeddings[self.word_vocab.index(word)]
            embeddings.append(dy.concatenate([tag_embedding, word_embedding]))

        if self.lstm_type == 'truncated':
            span_encoding = self.get_truncated_span_encoding_batch(embeddings, self.lstm_context_size)
        elif self.lstm_type == 'shuffled':
            span_encoding = self.get_shuffled_span_encoding_batch(embeddings, self.lstm_context_size)
        elif self.lstm_type == 'inside':
            span_encoding = self.get_inside_span_encoding(embeddings, self.lstm_context_size)
        else:
            span_encoding = self.get_basic_span_encoding(embeddings)

        def helper(left, right):
            assert 0 <= left < right <= len(sentence)

            label_scores = self.f_label(span_encoding(left, right))

            if is_train:
                oracle_label = gold.oracle_label(left, right)
                oracle_label_index = self.label_vocab.index(oracle_label)
                label_scores = augment(label_scores, oracle_label_index)

            label_scores_np = label_scores.npvalue()
            argmax_label_index = int(
                label_scores_np.argmax() if right - left < len(sentence) else
                label_scores_np[1:].argmax() + 1)
            argmax_label = self.label_vocab.value(argmax_label_index)

            if is_train:
                label = argmax_label if explore else oracle_label
                label_loss = (
                    label_scores[argmax_label_index] -
                    label_scores[oracle_label_index]
                    if argmax_label != oracle_label else dy.zeros(1))
            else:
                label = argmax_label
                label_loss = label_scores[argmax_label_index]

            if right - left == 1:
                tag, word = sentence[left]
                tree = trees.LeafParseNode(left, tag, word)
                if label:
                    tree = trees.InternalParseNode(label, [tree])
                return [tree], label_loss

            left_encodings = []
            right_encodings = []
            for split in range(left + 1, right):
                left_encodings.append(span_encoding(left, split))
                right_encodings.append(span_encoding(split, right))
            left_scores = self.f_split(dy.concatenate_to_batch(left_encodings))
            right_scores = self.f_split(dy.concatenate_to_batch(right_encodings))
            split_scores = left_scores + right_scores
            split_scores = dy.reshape(split_scores, (len(left_encodings),))

            if is_train:
                oracle_splits = gold.oracle_split(left, right)
                oracle_split = min(oracle_splits)
                oracle_split_index = oracle_split - (left + 1)
                split_scores = augment(split_scores, oracle_split_index)

            split_scores_np = split_scores.npvalue()
            argmax_split_index = int(split_scores_np.argmax())
            argmax_split = argmax_split_index + (left + 1)

            if is_train:
                split = argmax_split if explore else oracle_split
                split_loss = (
                    split_scores[argmax_split_index] -
                    split_scores[oracle_split_index]
                    if argmax_split != oracle_split else dy.zeros(1))
            else:
                split = argmax_split
                split_loss = split_scores[argmax_split_index]

            left_trees, left_loss = helper(left, split)
            right_trees, right_loss = helper(split, right)

            children = left_trees + right_trees
            if label:
                children = [trees.InternalParseNode(label, children)]

            return children, label_loss + split_loss + left_loss + right_loss

        children, loss = helper(0, len(sentence))
        assert len(children) == 1
        tree = children[0]
        if is_train and not explore:
            assert gold.convert().linearize() == tree.convert().linearize()
        return tree, loss
