import functools
import random

import dynet as dy
import numpy as np

import trees

START = "<START>"
STOP = "<STOP>"
UNK = "<UNK>"
COMMON_WORD = "<COMMON_WORD>"

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

def bow_range(items, start, end):
    if end <= start:
        return dy.zeros(*items[0].dim())
    else:
        return dy.average(items[start:end]) # or esum

def weighted_bow_range(items, start, end, params, location):
    if end <= start:
        return dy.zeros(*items[0].dim())
    else:
        selected_items = items[start:end]
        n = len(selected_items)
        selected_params = params[:n]
        assert location in ['left','right','middle']
        if location == 'left': # reverse weights to left
            selected_params = reversed(selected_params)
        elif location == 'middle' and n > 1: # mirror weights in middle
            first_half = selected_params[:n//2]
            selected_params[-(n//2):] = reversed(first_half)
            assert len(selected_params) == len(selected_items)
        weighted_items = [i*dy.parameter(p) for i, p in zip(selected_items, selected_params)]
        return dy.average(weighted_items)

class Feedforward(object):
    def __init__(self, model, input_dim, hidden_dims, output_dim, dropout=0):
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

        self.dropout = dropout

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
            x = dy.dropout(x, self.dropout)
        return x

class UntiedLSTMLayer(object):
    def __init__(self, model, in_size, hidden_size, length, dropout=0.0):
        self.model = model.add_subcollection("UntiedLSTM")

        self.in_size = in_size
        self.hidden_size = hidden_size
        self.length = length
        self.dropout = dropout

        self.Wxs = [self.model.add_parameters((4*hidden_size,in_size)) for _ in range(length)]
        self.Whs = [self.model.add_parameters((4*hidden_size,hidden_size)) for _ in range(length)]
        self.bs = [self.model.add_parameters(4*hidden_size) for _ in range(length)]
        self.initial_c = self.model.add_parameters(hidden_size)
        self.initial_h = self.model.add_parameters(hidden_size)

    def set_dropout(self, dropout):
        self.dropout = dropout

    def disable_dropout(self):
        self.dropout = 0.0

    def transduce(self, inputs):
        assert len(inputs) == self.length

        batch_size = inputs[0].dim()[1]

        dropout_retain = 1 - self.dropout
        dropout_mask_x = dy.random_bernoulli(self.in_size, dropout_retain, 1/dropout_retain, batch_size)
        dropout_mask_h = dy.random_bernoulli(self.hidden_size, dropout_retain, 1/dropout_retain, batch_size)

        c_init = dy.parameter(self.initial_c)
        c_tm1 = dy.concatenate_to_batch([c_init for _ in range(batch_size)])
        h_init = dy.parameter(self.initial_h)
        h_tm1 = dy.concatenate_to_batch([h_init for _ in range(batch_size)])

        outputs = []

        for i, x in enumerate(inputs):
            gates = dy.vanilla_lstm_gates_dropout_concat([x], h_tm1,
                    dy.parameter(self.Wxs[i]), dy.parameter(self.Whs[i]), dy.parameter(self.bs[i]),
                    dropout_mask_x, dropout_mask_h)

            c = dy.vanilla_lstm_c(c_tm1, gates)
            h = dy.vanilla_lstm_h(c, gates)
            outputs.append(h)

            c_tm1 = c
            h_tm1 = h

        return outputs

class BidirectionalUntiedLSTM(object):
    def __init__(self, model, in_size, hidden_size, n_layers, length, dropout=0.0):
        self.model = model.add_subcollection("Bidirectional")

        self.layers = []
        for i in range(n_layers):
            f = UntiedLSTMLayer(self.model, in_size if i == 0 else 2*hidden_size, hidden_size, length, dropout)
            b = UntiedLSTMLayer(self.model, in_size if i == 0 else 2*hidden_size, hidden_size, length, dropout)
            self.layers.append((f,b))

    def set_dropout(self, dropout):
        for f,b in self.layers:
            f.set_dropout(dropout)
            b.set_dropout(dropout)

    def disable_dropout(self):
        for f,b in self.layers:
            f.disable_dropout()
            b.disable_dropout()

    def transduce(self, inputs):
        f,b = self.layers[0]
        fh = f.transduce(inputs)
        bh = reversed(b.transduce(inputs[::-1]))
        h = [dy.concatenate([a,b]) for a,b in zip(fh, bh)]

        for i in range(1,len(self.layers)):
            f,b = self.layers[i]
            fh = f.transduce(h)
            bh = reversed(b.transduce(h[::-1]))
            h = [dy.concatenate([a,b]) for a,b in zip(fh, bh)]

        return h

class ParserBase(object):
    def __init__(
            self,
            model,
            tag_vocab,
            char_vocab,
            word_vocab,
            label_vocab,
            tag_embedding_dim,
            char_embedding_dim,
            char_lstm_layers,
            char_lstm_dim,
            word_embedding_dim,
            lstm_layers,
            lstm_dim,
            dropout,
            lstm_type,
            lstm_context_size,
            embedding_type,
            concat_bow,
            weight_bow,
            random_emb,
            random_lstm,
            common_word_threshold,
            no_lstm_hidden_dims,
    ):
        self.spec = locals()
        self.spec.pop("self")
        self.spec.pop("model")

        self.model = model.add_subcollection("Parser")
        self.trainable_parameters = self.model.add_subcollection("Trainable")
        self.tag_vocab = tag_vocab
        self.char_vocab = char_vocab
        self.word_vocab = word_vocab
        self.label_vocab = label_vocab
        self.char_lstm_dim = char_lstm_dim
        self.lstm_dim = lstm_dim

        emb_model = self.model if random_emb else self.trainable_parameters

        for c in embedding_type:
            assert c in 'wtc'
        self.embedding_type = embedding_type
        emb_dim = 0
        if 'w' in embedding_type:
            emb_dim += word_embedding_dim
            self.word_embeddings = emb_model.add_lookup_parameters(
                (word_vocab.size, word_embedding_dim), name="word-embeddings")
        if 't' in embedding_type:
            emb_dim += tag_embedding_dim
            self.tag_embeddings = emb_model.add_lookup_parameters(
                (tag_vocab.size, tag_embedding_dim), name="tag-embeddings")
        if 'c' in embedding_type:
            emb_dim += 2*char_lstm_dim
            self.char_embeddings = emb_model.add_lookup_parameters(
                (char_vocab.size, char_embedding_dim), name="char-embeddings")

            self.char_lstm = dy.BiRNNBuilder(
                char_lstm_layers,
                char_embedding_dim,
                2 * char_lstm_dim,
                self.trainable_parameters,
                dy.VanillaLSTMBuilder)

        if lstm_type in ["truncated", "untied-truncated", "no-lstm"]:
            self.indexed_starts = [self.trainable_parameters.add_parameters(emb_dim) for _ in range(300)]
            self.indexed_stops = [self.trainable_parameters.add_parameters(emb_dim) for _ in range(300)]

        if lstm_type == "no-lstm":
            self.context_network = Feedforward(
                self.model if random_lstm else self.trainable_parameters,
                emb_dim*2*(lstm_context_size+1), no_lstm_hidden_dims, 2*lstm_dim, dropout)
        elif lstm_type == "untied-truncated":
            self.lstm = BidirectionalUntiedLSTM(
                self.model if random_lstm else self.trainable_parameters,
                emb_dim, lstm_dim, lstm_layers, 2*(lstm_context_size+1))
        else:
            self.lstm = dy.BiRNNBuilder(
                lstm_layers,
                emb_dim,
                2 * lstm_dim,
                self.model if random_lstm else self.trainable_parameters,
                dy.VanillaLSTMBuilder)

        assert not (concat_bow and not lstm_type == 'truncated'), 'concat-bow only supported with truncated lstm-type'
        self.concat_bow = concat_bow
        self.weight_bow = weight_bow
        output_dim = 2 * lstm_dim
        if concat_bow:
            output_dim += 3 * emb_dim
            if weight_bow:
                self.bow_weights = [self.trainable_parameters.add_parameters(1) for i in range(300)]
        self.span_representation_dimension = output_dim

        self.dropout = dropout

        self.lstm_type = lstm_type
        self.lstm_context_size = lstm_context_size

        self.lstm_initialized = False

        self.common_word_threshold = common_word_threshold

    def param_collection(self):
        return self.model

    @classmethod
    def from_spec(cls, spec, model):
        return cls(model, **spec)

    def new_batch(self):
        self.lstm_initialized = False

    def transduce_lstm_batch(self, inputs):
        # this is a workaround for lstm dropout error in dynet
        if self.lstm_initialized:
            batch_size = inputs[0].dim()[1]
            for fb, bb in self.lstm.builder_layers:
                for b in [fb,bb]:
                    b.set_dropout_masks(batch_size=batch_size)
        self.lstm_initialized = True
        return self.lstm.transduce(inputs)

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

    def get_truncated_span_encoding(self, embeddings, distance, concat_bow, weight_bow, untied=False):
        padded_embeddings = [embeddings[0]]*(distance-1)+embeddings+[embeddings[-1]]*(distance-1)
        batched_embeddings = []
        batched_embeddings.append(dy.concatenate_to_batch([dy.parameter(p) for p in
                    self.indexed_starts[:len(embeddings)-1]]))
        for i in range(distance*2):
            selected = padded_embeddings[i:len(padded_embeddings)-(distance*2)+i+1]
            catted = dy.concatenate_to_batch(selected)
            batched_embeddings.append(catted)
        batched_embeddings.append(dy.concatenate_to_batch([dy.parameter(p) for p in
                    self.indexed_stops[:len(embeddings)-1]]))
        assert batched_embeddings[0].dim()[1] == len(embeddings)-1 # batch dimension is length of sentence + 1

        if untied:
            lstm_outputs = self.lstm.transduce(batched_embeddings)
        else:
            lstm_outputs = self.transduce_lstm_batch(batched_embeddings)

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

            if concat_bow:
                if weight_bow:
                    bow_before = weighted_bow_range(embeddings, 1, left-distance+1,
                            self.bow_weights, 'left')
                    bow_inside = weighted_bow_range(embeddings, left+distance+1, right-distance+1,
                            self.bow_weights, 'middle')
                    bow_after = weighted_bow_range(embeddings, right+distance+1, len(embeddings)-1,
                            self.bow_weights, 'right')
                else:
                    bow_before = bow_range(embeddings, 1, left-distance+1)
                    bow_inside = bow_range(embeddings, left+distance+1, right-distance+1)
                    bow_after = bow_range(embeddings, right+distance+1, len(embeddings)-1)
                return dy.concatenate([forward, backward, bow_before, bow_inside, bow_after])
            else:
                return dy.concatenate([forward, backward])

        return span_encoding

    def get_truncated_no_lstm_span_encoding(self, embeddings, distance):
        padded_embeddings = [embeddings[0]]*(distance-1)+embeddings+[embeddings[-1]]*(distance-1)
        batched_embeddings = []
        batched_embeddings.append(dy.concatenate_to_batch([dy.parameter(p) for p in
                    self.indexed_starts[:len(embeddings)-1]]))
        for i in range(distance*2):
            selected = padded_embeddings[i:len(padded_embeddings)-(distance*2)+i+1]
            catted = dy.concatenate_to_batch(selected)
            batched_embeddings.append(catted)
        batched_embeddings.append(dy.concatenate_to_batch([dy.parameter(p) for p in
                    self.indexed_stops[:len(embeddings)-1]]))
        assert batched_embeddings[0].dim()[1] == len(embeddings)-1 # batch dimension is length of sentence + 1

        context_outputs = self.context_network(dy.concatenate(batched_embeddings))

        @functools.lru_cache(maxsize=None)
        def span_encoding(left, right):
            return dy.pick_batch_elem(context_outputs, right) - \
                   dy.pick_batch_elem(context_outputs, left)

        return span_encoding

    def get_shuffled_span_encoding(self, embeddings, distance):
        all_lstm_inputs = []
        for i in range(len(embeddings)-1):
            lstm_inputs = embeddings[:] # copy
            shuffle(lstm_inputs, 1, i-distance+1) # note not shuffling start/end padding
            shuffle(lstm_inputs, i+distance+1, len(embeddings)-1)
            all_lstm_inputs.append(lstm_inputs)

        all_lstm_inputs = [dy.concatenate_to_batch(items) for items in transpose_lists(all_lstm_inputs)]
        all_lstm_outputs = self.transduce_lstm_batch(all_lstm_inputs)

        @functools.lru_cache(maxsize=None)
        def span_encoding(left, right):
            forward = (
                dy.pick_batch_elem(all_lstm_outputs[right], right)[:self.lstm_dim] -
                dy.pick_batch_elem(all_lstm_outputs[left], left)[:self.lstm_dim])
            backward = (
                dy.pick_batch_elem(all_lstm_outputs[left + 1], left)[self.lstm_dim:] -
                dy.pick_batch_elem(all_lstm_outputs[right + 1], right)[self.lstm_dim:])
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

    def get_embeddings(self, sentence, is_train=False):
        embeddings = []
        for tag, word in [(START, START)] + sentence + [(STOP, STOP)]:
            embed = []
            is_common_word = word not in (START,STOP) and \
                             self.word_vocab.count(word) > self.common_word_threshold
            if 't' in self.embedding_type:
                if is_common_word:
                    tag = COMMON_WORD
                tag_embedding = self.tag_embeddings[self.tag_vocab.index(tag)]
                embed.append(tag_embedding)
            if 'c' in self.embedding_type:
                chars = list(word) if word not in (START, STOP) else [word]
                if is_common_word:
                    chars = [COMMON_WORD]
                char_lstm_outputs = self.char_lstm.transduce([
                    self.char_embeddings[self.char_vocab.index(char)]
                    for char in [START] + chars + [STOP]])
                char_encoding = dy.concatenate([
                    char_lstm_outputs[-1][:self.char_lstm_dim],
                    char_lstm_outputs[0][self.char_lstm_dim:]])
                embed.append(char_encoding)
            if 'w' in self.embedding_type:
                if word not in (START, STOP):
                    count = self.word_vocab.count(word)
                    if not count or (is_train and np.random.rand() < 1 / (1 + count)):
                        word = UNK
                word_embedding = self.word_embeddings[self.word_vocab.index(word)]
                embed.append(word_embedding)
            embeddings.append(dy.concatenate(embed))
        return embeddings

    def get_representation_function(self, sentence, is_train):
        if self.lstm_type != "no-lstm":
            if is_train:
                self.lstm.set_dropout(self.dropout)
            else:
                self.lstm.disable_dropout()
        if 'c' in self.embedding_type:
            if is_train:
                self.char_lstm.set_dropout(self.dropout)
            else:
                self.char_lstm.disable_dropout()


        embeddings = self.get_embeddings(sentence, is_train)

        if self.lstm_type == 'truncated' or self.lstm_type == 'untied-truncated':
            span_encoding = self.get_truncated_span_encoding(embeddings, self.lstm_context_size, self.concat_bow, self.weight_bow, self.lstm_type == 'untied-truncated')
        elif self.lstm_type == 'no-lstm':
            span_encoding = self.get_truncated_no_lstm_span_encoding(embeddings, self.lstm_context_size)
        elif self.lstm_type == 'shuffled':
            span_encoding = self.get_shuffled_span_encoding(embeddings, self.lstm_context_size)
        elif self.lstm_type == 'inside':
            span_encoding = self.get_inside_span_encoding(embeddings, self.lstm_context_size)
        else:
            span_encoding = self.get_basic_span_encoding(embeddings)

        return span_encoding

    def lstm_derivative(self, sentence, position, index):
        self.lstm.disable_dropout()
        embeddings = self.get_embeddings(sentence, is_train=False)
        lstm_outputs = self.lstm.transduce(embeddings)

        forward = lstm_outputs[position][:self.lstm_dim]
        backward = lstm_outputs[position + 1][self.lstm_dim:]
        c = dy.concatenate([forward, backward])
        s = c[index]
        s.backward()
        gradients = [embed.gradient() for embed in embeddings]
        return gradients

class TopDownParser(ParserBase):
    def __init__(
            self,
            model,
            label_hidden_dim,
            split_hidden_dim,
            span_representation_args
    ):
        super().__init__(model, *span_representation_args)

        self.spec = {'label_hidden_dim':label_hidden_dim, 'split_hidden_dim':split_hidden_dim, 'span_representation_args':span_representation_args}

        self.f_label = Feedforward(
            self.trainable_parameters, self.span_representation_dimension, [label_hidden_dim], self.label_vocab.size)
        self.f_split = Feedforward(
            self.trainable_parameters, self.span_representation_dimension, [split_hidden_dim], 1)

    def parse(self, sentence, gold=None, explore=True):
        is_train = gold is not None

        get_span_encoding = self.get_representation_function(sentence, is_train)

        def helper(left, right):
            assert 0 <= left < right <= len(sentence)

            label_scores = self.f_label(get_span_encoding(left, right))

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
                left_encodings.append(get_span_encoding(left, split))
                right_encodings.append(get_span_encoding(split, right))
            left_scores = self.f_split(dy.concatenate_to_batch(left_encodings))
            right_scores = self.f_split(dy.concatenate_to_batch(right_encodings))
            split_scores = left_scores + right_scores
            split_scores = dy.reshape(split_scores, (len(left_encodings),))

            if is_train:
                oracle_splits = gold.oracle_splits(left, right)
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

class ChartParser(ParserBase):
    def __init__(
            self,
            model,
            label_hidden_dim,
            span_representation_args
    ):
        super().__init__(model, *span_representation_args)

        self.spec = {'label_hidden_dim':label_hidden_dim, 'span_representation_args':span_representation_args}

        self.f_label = Feedforward(
            self.trainable_parameters, self.span_representation_dimension, [label_hidden_dim], self.label_vocab.size - 1)

    def parse(self, sentence, gold=None):
        is_train = gold is not None

        get_span_encoding = self.get_representation_function(sentence, is_train)

        @functools.lru_cache(maxsize=None)
        def get_label_scores(left, right):
            non_empty_label_scores = self.f_label(get_span_encoding(left, right))
            return dy.concatenate([dy.zeros(1), non_empty_label_scores])

        def helper(force_gold):
            if force_gold:
                assert is_train

            chart = {}

            for length in range(1, len(sentence) + 1):
                for left in range(0, len(sentence) + 1 - length):
                    right = left + length

                    label_scores_expr = get_label_scores(left, right)
                    label_scores_np = label_scores_expr.npvalue()

                    if is_train:
                        oracle_label = gold.oracle_label(left, right)
                        oracle_label_index = self.label_vocab.index(oracle_label)

                    if force_gold:
                        label = oracle_label
                        label_score_expr = label_scores_expr[oracle_label_index]
                        label_score = label_scores_np[oracle_label_index]
                    else:
                        if is_train:
                            # augment the np version, which we use to get argmax
                            # the _expr versions won't have augmentation, but derivative is same
                            label_scores_np += 1
                            label_scores_np[oracle_label_index] -= 1
                        argmax_label_index = int(
                            label_scores_np.argmax() if length < len(sentence) else
                            label_scores_np[1:].argmax() + 1)
                        argmax_label = self.label_vocab.value(argmax_label_index)
                        label = argmax_label
                        label_score_expr = label_scores_expr[argmax_label_index]
                        label_score = label_scores_np[argmax_label_index]

                    if length == 1:
                        tag, word = sentence[left]
                        tree = trees.LeafParseNode(left, tag, word)
                        if label:
                            tree = trees.InternalParseNode(label, [tree])
                        chart[left, right] = [tree], label_score, label_score_expr
                        continue

                    if force_gold:
                        oracle_splits = gold.oracle_splits(left, right)
                        oracle_split = min(oracle_splits)
                        best_split = oracle_split
                    else:
                        best_split = max(
                            range(left + 1, right),
                            key=lambda split:
                                chart[left, split][1] +
                                chart[split, right][1])

                    left_trees, left_score, left_score_expr = chart[left, best_split]
                    right_trees, right_score, right_score_expr = chart[best_split, right]

                    children = left_trees + right_trees
                    if label:
                        children = [trees.InternalParseNode(label, children)]

                    chart[left, right] = (children, label_score + left_score + right_score,
                        label_score_expr + left_score_expr + right_score_expr)

            children, score, score_expr = chart[0, len(sentence)]
            assert len(children) == 1
            return children[0], score, score_expr

        tree, score, score_expr = helper(False)
        if is_train:
            oracle_tree, oracle_score, oracle_score_expr = helper(True)
            assert oracle_tree.convert().linearize() == gold.convert().linearize()
            correct = tree.convert().linearize() == gold.convert().linearize()
            loss_expr = dy.zeros(1) if correct else score_expr - oracle_score_expr
            loss = 0 if correct else score - oracle_score
            augmentation = loss - loss_expr.value()
            return tree, loss_expr + augmentation
        else:
            return tree, score_expr

class IndependentParser(ParserBase):
    def __init__(
            self,
            model,
            label_hidden_dim,
            span_representation_args
    ):
        super().__init__(model, *span_representation_args)

        self.spec = {'label_hidden_dim':label_hidden_dim, 'span_representation_args':span_representation_args}

        self.f_label = Feedforward(
            self.trainable_parameters, self.span_representation_dimension, [label_hidden_dim], self.label_vocab.size - 1)

    def parse(self, sentence, gold=None):
        is_train = gold is not None

        get_span_encoding = self.get_representation_function(sentence, is_train)

        @functools.lru_cache(maxsize=None)
        def get_label_scores(left, right):
            non_empty_label_scores = self.f_label(get_span_encoding(left, right))
            return dy.concatenate([dy.zeros(1), non_empty_label_scores])

        brackets = trees.SpanList(sentence)
        total_loss = dy.zeros(1)
        for length in range(1, len(sentence) + 1):
            for left in range(0, len(sentence) + 1 - length):
                right = left + length

                label_scores_expr = get_label_scores(left, right)
                label_scores_np = label_scores_expr.npvalue()

                if is_train:
                    oracle_label = gold.oracle_label(left, right)
                    oracle_label_index = self.label_vocab.index(oracle_label)
                    oracle_label_score_expr = label_scores_expr[oracle_label_index]

                    # augment the np version, which we use to get argmax
                    # the _expr versions won't have augmentation, but derivative is same
                    label_scores_np += 1
                    label_scores_np[oracle_label_index] -= 1

                argmax_label_index = int(
                    label_scores_np.argmax() if length < len(sentence) else
                    label_scores_np[1:].argmax() + 1)
                argmax_label = self.label_vocab.value(argmax_label_index)
                label = argmax_label
                label_score_expr = label_scores_expr[argmax_label_index]
                label_score = label_scores_np[argmax_label_index]
                for sublabel in label: # note that no_label is just an empty tuple
                    brackets.add(left, right, sublabel)

                if is_train and argmax_label != oracle_label:
                    total_loss = total_loss + label_score_expr - oracle_label_score_expr

        return brackets, total_loss

class LabelPrediction(ParserBase):
    def __init__(
            self,
            model,
            parser,
            label_hidden_dim,
    ):

        self.parser = parser
        self.label_hidden_dim = label_hidden_dim
        self.f_label = Feedforward(
            model, parser.span_representation_dimension, [label_hidden_dim], parser.label_vocab.size)

    def predict_parent_label_for_spans(self, sentence, gold, self_not_parent=False):
        span_encoding = self.parser.get_representation_function(sentence, is_train=False)

        correct = 0
        total = 0
        total_loss = dy.zeros(1)
        def accumulate(left, right, target_label_index):
            nonlocal correct, total, total_loss
            label_scores = self.f_label(span_encoding(left, right))

            # predicted label
            label_scores_np = label_scores.npvalue()
            argmax_label_index = int(label_scores_np.argmax())
            if argmax_label_index == target_label_index:
                correct += 1
            total += 1

            # loss for training
            augmented_label_scores = augment(label_scores, target_label_index)
            augmented_argmax_label_index = int(augmented_label_scores.npvalue().argmax())
            label_loss = (
                label_scores[augmented_argmax_label_index] -
                label_scores[target_label_index]
                if augmented_argmax_label_index != target_label_index else dy.zeros(1))
            total_loss = total_loss + label_loss

        for node, parent in gold.iterate_spans_with_parents(): # doesn't include top level
            label = node.label if self_not_parent else parent.label
            label_index = self.parser.label_vocab.index(label)
            accumulate(node.left, node.right, label_index)
        label = gold.label if self_not_parent else () # () represents no-label, since root has no parent
        label_index = self.parser.label_vocab.index(label)
        accumulate(gold.left, gold.right, label_index)

        return total_loss, correct, total
