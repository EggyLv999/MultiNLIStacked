# Stacked BiLSTMs for MultiNLI
Modular re-implementation of Nie and Bansal 2017, https://arxiv.org/abs/1708.02312 the current State of the Art for the Natural Language Inference task on sentence embeddings. This task involves deciding the relationship between two sentences as entailment (E), contradiction (C), or neutral (N). Examples:

John wrote a report, and Bill said Peter did too. -> Bill said Peter wrote a report. (Entailment)

John loves cats. -> Mary loves dogs. (Neutral)

No delegate finished the report. -> Some delegate finished the report on time. (Contradiction)

There is one additional constraint, which is that the model must produce a sentence embedding, which is a fixed-length vector, for each sentence separately before deciding their relationship. The full list of rules we followed can be found at the (RepEval 2017)[https://repeval2017.github.io/] workshop page.

We made some improvements on the MultiNLI system, but they cannot be open-sourced quite yet. This is a limited copy which still implements the original state of the art system.
