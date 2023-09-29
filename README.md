# Subtopic-oriented biomedical extractive summarization
Summarization framework based on subtopic extraction and
extractive summarization using pretrained language models.


## Framework evaluation

### Installation
Install the dependencies by running:
```
pip install -r framework/requirements.txt
```

### Dataset preprocessing
Refer to [biomed-ext-summ](https://github.com/NotXia/biomed-ext-summ#preprocessing)
for converting an abstractive dataset into an extractive one.


### Evaluation
To evaluate a configuration of the framework, run:
```
python eval/eval.py                             \
    --dataset=<dataset name>                    \
    --dataset-path=<path to extractive dataset> \
    --embedding=<embedding>
```
Available `<embedding>` are `bow`, `word2vec`, `glove`, `fasttext`, `biowordvec`, 
`minilm`, `biobert` and `pubmedbert`.

Instead of `--embedding`, the following options are available:
- `--plain` to evaluate using only a pretrained model (without the framework).
- `--oracle` to evaluate using the reference extractive summary.


## Web app prototype

This is a prototype to present the result of the framework to the end user.

To start the application, run in the repository root:
```
docker compose up
```
By default, the application will be served at http://localhost:8001.

Note that [nvidia-container-toolkit](https://github.com/NVIDIA/nvidia-docker) 
is required in order to use the GPU in a container.