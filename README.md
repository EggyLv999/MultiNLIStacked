# Stacked BiLSTMs for MultiNLI
Modular re-implementation of Nie and Bansal 2017, https://arxiv.org/abs/1708.02312 the current state of the art for the Natural Language Inference task on sentence embeddings. This task involves deciding the relationship between two sentences as entailment (E), contradiction (C), or neutral (N). Examples:

John wrote a report, and Bill said Peter did too. -> Bill said Peter wrote a report. (Entailment)

John loves cats. -> Mary loves dogs. (Neutral)

No delegate finished the report. -> Some delegate finished the report on time. (Contradiction)

There is one additional constraint, which is that the model must produce a sentence embedding, which is a fixed-length vector, for each sentence separately before deciding their relationship. The full list of rules we followed can be found at the [RepEval 2017](https://repeval2017.github.io/) workshop page.

This was joint work with Paul Michel. We made some improvements on the MultiNLI system, but they cannot be open-sourced quite yet. This is a limited copy which still implements the original state of the art system.

## Getting the data

There are two major NLI corpora, the [SNLI](https://nlp.stanford.edu/projects/snli/) and [MultiNLI](https://www.nyu.edu/projects/bowman/multinli/).
Ideally they should be combined for training

### SNLI

The version included right now is intented for testing (only has 10,000 training samples)

Download and unzip with these commands:

```bash
wget https://nlp.stanford.edu/projects/snli/snli_1.0.zip
unzip snli_1.0.zip
```

### MultiNLI

Download and unzip with these commands:

```bash
wget https://www.nyu.edu/projects/bowman/multinli/multinli_1.0.zip
unzip snli_1.0.zip
```

## Running experiments

First install (dynn)[https://github.com/pmichel31415/dynn].

The configuration for the experiments can be specified via the command line (see `options.py`) or in a `.yaml` file (see `config/snli_test.yaml` for example).

you can run any experiment with:

```bash
python nli.py --config_file [config_file.yaml] --env [train, test]
```

The `--env` argument allows you to specify a subset of options in a yaml file (typically `test`, `train`, `tune`...). This way some options are shared accross environments.

To launch an experiment on the cluster, write a `.sh` file on the model of `experiments/run_snli.sh`, then launch

```bash
sbatch [-o log_file.txt] experiments/run_snli.sh
```

## Pretrained word vectors

Most NLI systems start from the 300D glove vectors trained on 840B words. The file can be found here :

```bash
wget http://nlp.stanford.edu/data/glove.840B.300d.zip
unzip glove.840B.300d.zip
```

Once the file is unzipped, it needs to be preprocessed before it's used for our system. Assume we have a vocabulary file `vocab_file` created with a previous run of our system, just run:

```bash
python scripts/glove_to_numpy.py /path/to/glove.840B.300d.txt vocab_file output_embeddings 300
```

`output_embeddings` now contains a numpy array of the right dimensions and can be fed to our model with `--pretrained_wembs`.
Note that for now the words that aren't found in the glove file are initialized at 0 (should maybe change this to random init).

