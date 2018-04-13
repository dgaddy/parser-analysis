# Neural Constituency Parser Analysis

This repository contains code necessary to reproduce experiments in *What's Going On in Neural Constituency Parsers? An Analysis* from NAACL 2018.

If you are looking for a parser implementation and not the analysis, we recommend you instead use the code from [Mitchell's repository](https://github.com/mitchellstern/minimal-span-parser), which also includes the model improvements described in the paper.

## Requirements and Setup

* Python 3.5 or higher.
* [DyNet](https://github.com/clab/dynet). We recommend installing DyNet from source with MKL support for significantly faster run time.
* [EVALB](http://nlp.cs.nyu.edu/evalb/). Before starting, run `make` inside the `EVALB/` directory to compile an `evalb` executable. This will be called from Python for evaluation.

## Command Line Arguments

The base model can be trained with the command:
```
python3 src/main.py train --parser-type chart --model-path-base models/base-model
```
The dev score will be appended to the model file name in the form `_dev=xx.xx`, where each `x` is replaced with a digit, so this will need to be specified when running the program with an already trained model as is done for some experiments.

The following table describes the command line arguments to run each experiment in the paper:

Paper section | Arguments
3.1 | Run `python3 src/main.py train-label --model-path-base models/base-model_dev=xx.xx`.
3.2 | Use the base model command with `--parser-type independent` instead of `chart`.
4.1 | Add the option `--embedding-type` with combinations of the characters w, t, and c for word, tag, and character (e.g. `--embedding-type wt`).  For character only, we recommend using `--char-lstm-dim 250` as well.
5.1 | Run `python3 src/main.py derivative --model-path-base models/base-model_dev=xx.xx`.
5.2 | Add `--lstm-type truncated --lstm-context-size 3` to the base model command and use different values for the context size.
5.3 | Add `--lstm-type shuffled --lstm-context-size 3`.
5.4 | Add `--lstm-type no-lstm --lstm-context-size 3 --no-lstm-hidden-dims 1000`.

To run on the test set, use
```
python3 src/main.py test --model-path-base models/base-model_dev=xx.xx
```

