import argparse
import itertools
import os.path
import time
from collections import defaultdict
import random

import dynet as dy
import numpy as np

import evaluate
import parse
import trees
import vocabulary

def format_elapsed(start_time):
    elapsed_time = int(time.time() - start_time)
    minutes, seconds = divmod(elapsed_time, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    elapsed_string = "{}h{:02}m{:02}s".format(hours, minutes, seconds)
    if days > 0:
        elapsed_string = "{}d{}".format(days, elapsed_string)
    return elapsed_string

def run_train(args):
    print("Running training with arguments:", args)

    if args.numpy_seed is not None:
        print("Setting numpy random seed to {}...".format(args.numpy_seed))
        np.random.seed(args.numpy_seed)

    print("Loading training trees from {}...".format(args.train_path))
    train_treebank = trees.load_trees(args.train_path)
    print("Loaded {:,} training examples.".format(len(train_treebank)))

    print("Loading development trees from {}...".format(args.dev_path))
    dev_treebank = trees.load_trees(args.dev_path)
    print("Loaded {:,} development examples.".format(len(dev_treebank)))

    print("Processing trees for training...")
    train_parse = [tree.convert() for tree in train_treebank]

    print("Constructing vocabularies...")

    tag_vocab = vocabulary.Vocabulary()
    tag_vocab.index(parse.START)
    tag_vocab.index(parse.STOP)
    tag_vocab.index(parse.COMMON_WORD)

    char_vocab = vocabulary.Vocabulary()
    char_vocab.index(parse.START)
    char_vocab.index(parse.STOP)
    char_vocab.index(parse.COMMON_WORD)
    char_vocab.index(parse.UNK)

    word_vocab = vocabulary.Vocabulary()
    word_vocab.index(parse.START)
    word_vocab.index(parse.STOP)
    word_vocab.index(parse.UNK)

    label_vocab = vocabulary.Vocabulary()
    label_vocab.index(())

    for tree in train_parse:
        nodes = [tree]
        while nodes:
            node = nodes.pop()
            if isinstance(node, trees.InternalParseNode):
                label_vocab.index(node.label)
                nodes.extend(reversed(node.children))
            else:
                tag_vocab.index(node.tag)
                word_vocab.index(node.word)
                for char in node.word:
                    char_vocab.index(char)

    tag_vocab.freeze()
    char_vocab.freeze()
    word_vocab.freeze()
    label_vocab.freeze()

    def print_vocabulary(name, vocab):
        special = {parse.START, parse.STOP, parse.UNK}
        print("{} ({:,}): {}".format(
            name, vocab.size,
            sorted(value for value in vocab.values if value in special) +
            sorted(value for value in vocab.values if value not in special)))

    if args.print_vocabs:
        print_vocabulary("Tag", tag_vocab)
        print_vocabulary("Word", word_vocab)
        print_vocabulary("Character", char_vocab)
        print_vocabulary("Label", label_vocab)

    print("Initializing model...")
    model = dy.ParameterCollection()
    print("Input LSTM type:", args.lstm_type)
    assert args.embedding_type != ""
    span_representation_args = [
        tag_vocab,
        char_vocab,
        word_vocab,
        label_vocab,
        args.tag_embedding_dim,
        args.char_embedding_dim,
        args.char_lstm_layers,
        args.char_lstm_dim,
        args.word_embedding_dim,
        args.lstm_layers,
        args.lstm_dim,
        args.dropout,
        args.lstm_type,
        args.lstm_context_size,
        args.embedding_type,
        args.concat_bow,
        args.weight_bow,
        args.random_embeddings,
        args.random_lstm,
        args.common_word_threshold,
        args.no_lstm_hidden_dims,
    ]
 
    if args.parser_type == "top-down":
        parser = parse.TopDownParser(
            model,
            args.label_hidden_dim,
            args.split_hidden_dim,
            span_representation_args
        )
    elif args.parser_type == 'chart':
        parser = parse.ChartParser(
            model,
            args.label_hidden_dim,
            span_representation_args
        )
    elif args.parser_type == 'independent':
        parser = parse.IndependentParser(
            model,
            args.label_hidden_dim,
            span_representation_args
        )
    trainer = dy.AdamTrainer(parser.trainable_parameters)

    total_processed = 0
    current_processed = 0
    check_every = len(train_parse) / args.checks_per_epoch
    best_dev_fscore = -np.inf
    best_dev_model_path = None

    start_time = time.time()

    def check_dev():
        nonlocal best_dev_fscore
        nonlocal best_dev_model_path

        dev_start_time = time.time()

        dev_predicted = []
        for tree in dev_treebank:
            dy.renew_cg()
            sentence = [(leaf.tag, leaf.word) for leaf in tree.leaves()]
            predicted, _ = parser.parse(sentence)
            dev_predicted.append(predicted.convert())

        if args.parser_type == 'independent':
            tree_count = 0
            for pred in dev_predicted:
                if pred.is_tree():
                    tree_count += 1
            print("Percentage of valid trees:", tree_count/len(dev_predicted))

            dev_fscore = evaluate.bracket_f1(dev_treebank, dev_predicted)
        else:
            dev_fscore = evaluate.evalb(args.evalb_dir, dev_treebank, dev_predicted)

        print(
            "dev-fscore {} "
            "dev-elapsed {} "
            "total-elapsed {}".format(
                dev_fscore,
                format_elapsed(dev_start_time),
                format_elapsed(start_time),
            )
        )

        if dev_fscore.fscore > best_dev_fscore:
            if best_dev_model_path is not None:
                for ext in [".data", ".meta"]:
                    path = best_dev_model_path + ext
                    if os.path.exists(path):
                        print("Removing previous model file {}...".format(path))
                        os.remove(path)

            best_dev_fscore = dev_fscore.fscore
            best_dev_model_path = "{}_dev={:.2f}".format(
                args.model_path_base, dev_fscore.fscore)
            print("Saving new best model to {}...".format(best_dev_model_path))
            dy.save(best_dev_model_path, [parser])

    for epoch in itertools.count(start=1):
        if args.epochs is not None and epoch > args.epochs:
            break

        np.random.shuffle(train_parse)
        epoch_start_time = time.time()

        for start_index in range(0, len(train_parse), args.batch_size):
            dy.renew_cg()
            parser.new_batch()
            batch_losses = []
            for tree in train_parse[start_index:start_index + args.batch_size]:
                sentence = [(leaf.tag, leaf.word) for leaf in tree.leaves()]
                if args.parser_type == "top-down":
                    _, loss = parser.parse(sentence, tree, args.explore)
                else:
                    _, loss = parser.parse(sentence, tree)
                batch_losses.append(loss)
                total_processed += 1
                current_processed += 1

            batch_loss = dy.average(batch_losses)
            batch_loss_value = batch_loss.scalar_value()
            batch_loss.backward()
            trainer.update()

            if (start_index // args.batch_size + 1) % args.print_frequency == 0:
                print(
                    "epoch {:,} "
                    "batch {:,}/{:,} "
                    "processed {:,} "
                    "batch-loss {:.4f} "
                    "epoch-elapsed {} "
                    "total-elapsed {}".format(
                        epoch,
                        start_index // args.batch_size + 1,
                        int(np.ceil(len(train_parse) / args.batch_size)),
                        total_processed,
                        batch_loss_value,
                        format_elapsed(epoch_start_time),
                        format_elapsed(start_time),
                    )
                )

            if current_processed >= check_every:
                current_processed -= check_every
                check_dev()

def run_test(args):
    print("Loading test trees from {}...".format(args.test_path))
    test_treebank = trees.load_trees(args.test_path)
    print("Loaded {:,} test examples.".format(len(test_treebank)))

    print("Loading model from {}...".format(args.model_path_base))
    model = dy.ParameterCollection()
    [parser] = dy.load(args.model_path_base, model)

    print("Parsing test sentences...")

    start_time = time.time()

    test_predicted = []
    for tree in test_treebank:
        dy.renew_cg()
        sentence = [(leaf.tag, leaf.word) for leaf in tree.leaves()]
        predicted, _ = parser.parse(sentence)
        test_predicted.append(predicted.convert())

    if type(parser) == parse.IndependentParser:
        print('Warning: not using evalb for evaluation')
        test_fscore = evaluate.bracket_f1(test_treebank, test_predicted)
    else:
        test_fscore = evaluate.evalb(args.evalb_dir, test_treebank, test_predicted)

    print(
        "test-fscore {} "
        "test-elapsed {}".format(
            test_fscore,
            format_elapsed(start_time),
        )
    )

def predict_labels(args):
    print("Loading training trees from {}...".format(args.train_path))
    train_treebank = trees.load_trees(args.train_path)
    print("Loaded {:,} training examples.".format(len(train_treebank)))

    print("Loading development trees from {}...".format(args.dev_path))
    dev_treebank = trees.load_trees(args.dev_path)
    print("Loaded {:,} development examples.".format(len(dev_treebank)))

    print("Processing trees for training...")
    train_parse = [tree.convert() for tree in train_treebank]
    dev_parse = [tree.convert() for tree in dev_treebank]

    print("Calculating baseline...")
    counts = defaultdict(lambda : defaultdict(int))
    for tree in train_parse:
        for node, parent in tree.iterate_spans_with_parents(): # doesn't include top level
            counts[node.label][parent.label] += 1
        counts[tree.label]['<NONE>'] += 1
    predictions = {label:max(counts.keys(), key=lambda x: counts[x]) for label, counts in counts.items()}
    correct = 0
    total = 0
    for tree in dev_parse:
        for node, parent in tree.iterate_spans_with_parents(): # doesn't include top level
            if predictions[node.label] == parent.label:
                correct += 1
            total += 1
        if predictions[tree.label] == '<NONE>':
            correct += 1
        total += 1
    print("baseline score:", correct/total)

    print("Loading model from {}...".format(args.model_path_base))
    model = dy.ParameterCollection()
    [base_parser] = dy.load(args.model_path_base, model)

    for self_not_parent in [False, True]:
        parser = parse.LabelPrediction(model, base_parser, args.label_hidden_dim)
        trainer = dy.AdamTrainer(parser.f_label.model)

        print('predicting own label' if self_not_parent else 'predicting parent label')
        for epoch_index in range(10):
            np.random.shuffle(train_parse)
            for start_index in range(0, len(train_parse), args.batch_size):
                dy.renew_cg()
                batch_losses = []
                for tree in train_parse[start_index:start_index + args.batch_size]:
                    sentence = [(leaf.tag, leaf.word) for leaf in tree.leaves()]
                    loss, _, _ = parser.predict_parent_label_for_spans(sentence, tree, self_not_parent)
                    batch_losses.append(loss)
                batch_loss = dy.average(batch_losses)
                batch_loss_value = batch_loss.scalar_value()
                batch_loss.backward()
                trainer.update()

            correct = 0
            total = 0
            for tree in dev_parse:
                dy.renew_cg()
                sentence = [(leaf.tag, leaf.word) for leaf in tree.leaves()]
                _, c, t = parser.predict_parent_label_for_spans(sentence, tree, self_not_parent)
                correct += c
                total += t
            print("dev score at epoch", epoch_index+1, ":", correct/total)

def derivative_analysis(args):
    print("Loading development trees from {}...".format(args.dev_path))
    dev_treebank = trees.load_trees(args.dev_path)
    print("Loaded {:,} development examples.".format(len(dev_treebank)))

    print("Processing trees...")
    dev_parse = [tree.convert() for tree in dev_treebank]

    print("Loading model from {}...".format(args.model_path_base))
    model = dy.ParameterCollection()
    [parser] = dy.load(args.model_path_base, model)

    total_l1_grad = np.zeros(500)
    total_l2_grad = np.zeros(500)
    total_count = np.zeros(500)
    for tree in dev_parse:
        sentence = [(leaf.tag, leaf.word) for leaf in tree.leaves()]
        for position in range(len(sentence)+1):
            index = random.randrange(parser.lstm_dim*2)
            dy.renew_cg()
            gradients = parser.lstm_derivative(sentence, position, index)
            buckets = list(reversed(range(position+1))) + list(range(len(sentence)-position+1))
            assert len(buckets) == len(gradients)
            for position, grad in zip(buckets, gradients):
                total_l1_grad[position] += np.linalg.norm(grad, ord=1)
                total_l2_grad[position] += np.linalg.norm(grad, ord=2)
                total_count[position] += 1

    print('l1:')
    for i in range(500):
        if total_count[i] == 0:
            break
        print(total_l1_grad[i]/total_count[i])
    print('l2:')
    for i in range(500):
        if total_count[i] == 0:
            break
        print(total_l2_grad[i]/total_count[i])

def main():
    dynet_args = [
        "--dynet-mem",
        "--dynet-weight-decay",
        "--dynet-autobatch",
        "--dynet-gpus",
        "--dynet-gpu",
        "--dynet-devices",
        "--dynet-seed",
    ]

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    subparser = subparsers.add_parser("train")
    subparser.set_defaults(callback=run_train)
    for arg in dynet_args:
        subparser.add_argument(arg)
    subparser.add_argument("--numpy-seed", type=int)
    subparser.add_argument("--parser-type", choices=["top-down", "chart", "independent"], required=True)
    subparser.add_argument("--tag-embedding-dim", type=int, default=50)
    subparser.add_argument("--char-embedding-dim", type=int, default=50)
    subparser.add_argument("--char-lstm-layers", type=int, default=1)
    subparser.add_argument("--char-lstm-dim", type=int, default=100)
    subparser.add_argument("--word-embedding-dim", type=int, default=100)
    subparser.add_argument("--lstm-layers", type=int, default=2)
    subparser.add_argument("--lstm-dim", type=int, default=250)
    subparser.add_argument("--label-hidden-dim", type=int, default=250)
    subparser.add_argument("--split-hidden-dim", type=int, default=250)
    subparser.add_argument("--dropout", type=float, default=0.4)
    subparser.add_argument("--explore", action="store_true")
    subparser.add_argument("--model-path-base", required=True)
    subparser.add_argument("--evalb-dir", default="EVALB/")
    subparser.add_argument("--train-path", default="data/02-21.10way.clean")
    subparser.add_argument("--dev-path", default="data/22.auto.clean")
    subparser.add_argument("--batch-size", type=int, default=10)
    subparser.add_argument("--epochs", type=int)
    subparser.add_argument("--checks-per-epoch", type=int, default=4)
    subparser.add_argument("--print-vocabs", action="store_true")
    subparser.add_argument("--lstm-type", choices=["basic","truncated","shuffled","inside","no-lstm","untied-truncated"], default="basic")
    subparser.add_argument("--lstm-context-size", type=int, default=3)
    subparser.add_argument("--embedding-type", default="wc") # characters w/t/c for word/tag/character
    subparser.add_argument("--random-embeddings", action="store_true")
    subparser.add_argument("--random-lstm", action="store_true")
    subparser.add_argument("--concat-bow", action="store_true")
    subparser.add_argument("--weight-bow", action="store_true")
    subparser.add_argument("--print-frequency", type=int, default=1)
    subparser.add_argument("--common-word-threshold", type=int, default=float('inf')) # replace tags and character-level inputs with a special token above this threshold
    subparser.add_argument("--no-lstm-hidden-dims", type=int, nargs="+", default=[250])
    train_subparser = subparser

    subparser = subparsers.add_parser("train-label", parents=[train_subparser], add_help=False)
    subparser.set_defaults(callback=predict_labels)

    subparser = subparsers.add_parser("derivative", parents=[train_subparser], add_help=False)
    subparser.set_defaults(callback=derivative_analysis)

    subparser = subparsers.add_parser("test")
    subparser.set_defaults(callback=run_test)
    for arg in dynet_args:
        subparser.add_argument(arg)
    subparser.add_argument("--model-path-base", required=True)
    subparser.add_argument("--evalb-dir", default="EVALB/")
    subparser.add_argument("--test-path", default="data/23.auto.clean")

    args = parser.parse_args()
    args.callback(args)

if __name__ == "__main__":
    main()
