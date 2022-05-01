## PCEE-BERT

This repository contains the code for our work [PCEE-BERT](https://openreview.net/pdf?id=K_fV_YHD_D).

**************************** **Updates** ****************************

<!-- Thanks for your interest in our repo! -->

<!-- Probably you will think this as another *"empty"* repo of a preprint paper 🥱.
Wait a minute! The authors are working day and night 💪, to make the code and models available, so you can explore our state-of-the-art sentence embeddings.
We anticipate the code will be out * **in a few weeks** *. -->

* 2022/05/01: upload repo; 


### datasets

We use the glue benchmark datasets, which are accessible online; 

### How to run

First, you should finetune a multi-exit BERT on a glue task, say MRPC: 

```bash
./scripts/mrpc/train.sh
```

Then, run inference with our PCEE-BERT early exiting mechanism, with different patience parameter and threshold parameter:
```bash
./scripts/mrpc/inference_ee_mechanism_V1.sh
```

### reference

TBD





