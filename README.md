# argument-reasoning-comprehension

This is the repository for the codes, targeting the task: SemEval 2018 Task12, the argument reasoning comprehension task.

## Task Introduction

* https://competitions.codalab.org/competitions/17327

* Select the correct `Warrant` that appropriately explain the given `Argument`(claim + reason).

## Reference

* Plan to try similar approach with previous top-score submission ([GIST](https://www.google.co.kr/url?sa=t&source=web&rct=j&url=http://aclweb.org/anthology/S18-1122&ved=2ahUKEwiWw5uS7-bdAhWBVbwKHVv_BlQQFjAAegQIARAB&usg=AOvVaw1l7GdyLiKN2PyXEEAN1tYy))

* Use pretrained sentence embedding model([Quan Chen et al. 2017](http://aclweb.org/anthology/W17-5307)), trained by larger dataset([MultiNLI](https://www.nyu.edu/projects/bowman/multinli/)).

* nyu-mll's ESIM model repository [[github]](https://github.com/nyu-mll/multiNLI/blob/master/python/models/esim.py)
