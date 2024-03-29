# argument-reasoning-comprehension

This is a repository for the codes, targeting a task: SemEval 2018 Task12, the argument reasoning comprehension task. Current version is the reimplementation of [this paper](http://aclweb.org/anthology/S18-1122) in Tensorflow.

## Task Introduction

* https://competitions.codalab.org/competitions/17327
* Select the correct `Warrant` that appropriately explains the given `Argument`(claim + reason).



## Getting Started

#### Requirements

- tensorflow-gpu==1.8
- numpy==1.16.2
- nltk==3.4.5
- stanfordcorenlp=3.9.1.1 (optional)



#### Prerequistes

1. Prepare the SNLI dataset and locate it to `/data/nli` directory.

   (Download:  [SNLI](<https://nlp.stanford.edu/projects/snli/>))

2. Prepare the shared task dataset and locate it to `data/main` directory.

   (Download:   [SharedTask](<https://github.com/UKPLab/argument-reasoning-comprehension-task/tree/master/mturk/annotation-task/data/exported-SemEval2018-train-dev-test>) )

3. Download Glove word embedding and locate it to `/data/embed` directory.

4. Locate the above resource into appropriate location, followed by the package structure below.



#### commands


Preprocess dataset
```
python preprocessing.py
```
Train ESIM model for transfer learning.
```
python script.py --mode=esim_train --model=esim
```
Train main model with pretrained ESIM model.
```
python script.py --mode=train --model=main
```
Evaluation 
```
python script.py --mode=eval --model=main
```


#### Package Structure

```
├── argument-reasoning-comprehension
│     └── script.py
│     └── data_helper.py
│     └── preprocessing.py
│     └── util.py
│     └── esim_model.py
│     └── model.py
│     └── data/
│     		└──── log/ (Trained models and logs are here.)
│     		└──── emb/
│ 		    		└──── glove.6B.300d.txt
│     		└──── main/
│ 		    		└──── train_full.txt
│ 		    		└──── dev_full.txt
│ 		    		└──── test_full.txt
│     		└──── nli/
│		     		└──── snli_1.0/
│				     		└──── snli_1.0.train.txt
│				     		└──── snli_1.0.dev.txt
│				     		└──── snli_1.0.test.txt
│     		└──── stanford_corenlp/
│			     		└──── (Unziped parser data)
```



#### Results

```
SNLI testset accuracy: 83.83%
Shared task testset accuracy: 56.53%
```



## References

* Choi and Lee, **GIST at SemEval-2018 Task 12: A network transferring inference knowledge to Argument Reasoning Comprehension task** (Semeval 2018 Task12) [[paper]](http://aclweb.org/anthology/S18-1122)

* Qian Chen et al., **Enhanced LSTM for Natural Language Inference** (ACL 2017) [[paper]](http://www.aclweb.org/anthology/P17-1152)

* Dataset to train Chen's sentence embedding model [[page]](https://www.nyu.edu/projects/bowman/multinli/) [[paper]](http://aclweb.org/anthology/N18-1101)

* nyu-mll's ESIM model repository [[github]](https://github.com/nyu-mll/multiNLI/blob/master/python/models/esim.py)
