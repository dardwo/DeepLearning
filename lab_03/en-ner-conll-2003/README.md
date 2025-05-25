CoNLL-2003 English Named Entity Recognition.
======================================================

NER challenge for CoNLL-2003 English.
Annotations were taken from [University of Antwerp](https://www.clips.uantwerpen.be/conll2003/ner/).
The English data is a collection of news wire articles from the [Reuters Corpus](https://trec.nist.gov/data/reuters/reuters.html), RCV1.

Format of the train set
-----------------------

The train set has just two columns separated by TABs:

* the expected BIO labels,
* the docuemnt.

Each line is a separate training item. Note that this is TSV format,
not CSV, double quotes are not interpreted in a special way!

Preprocessing snippet located [here](https://git.applica.pl/snippets/18)

End-of-lines inside documents were replaced with the '</S>' tag.

The train is compressed with the xz compressor, in order to see a
random sample of 10 training items, run:

    xzcat train/train.tsv.xz | shuf -n 10 | less -S

(The `-S` disables line wrapping, press "q" to exit `less` browser.)

Format of the test sets
-----------------------

For the test sets, the input data is given in two files: the text in
`in.tsv` and the expected labels in `expected.tsv`. (The files have
`.tsv` extensions for consistency but actually they do not contain TABs.)

To see the first 5 test items run:

    cat dev-0/in.tsv | paste dev-0/expected.tsv - | head -n 5

The `expected.tsv` file for the `test-A` test set is hidden and is not
available in the master branch.


Evaluation metrics
------------------

One evaluation metric is used:

* BIO-F1 

Directory structure
-------------------

* `README.md` — this file
* `config.txt` — GEval configuration file
* `train/` — directory with training data
* `train/train.tsv.xz` — train set
* `dev-0/` — directory with dev (test) data (split preserved from CoNLL-2003)
* `dev-0/in.tsv` — input data for the dev set
* `dev-0/expected.tsv` — expected (reference) data for the dev set
* `test-A` — directory with test data
* `test-A/in.tsv` — input data for the test set
* `test-A/expected.tsv` — expected (reference) data for the test set (hidden from the developers,
   not available in the `master` branch)

Usually, there is no reason to change these files.

