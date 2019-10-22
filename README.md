# Transformer based LM

This repository contains the code used for GPT, [Improving Language Understanding by Generative Pre-Training](https://arxiv.org/abs/1810.04805) by Alec Radford, Karthik Narasimhan, Tim Salimans and Ilya Sutskever.

Some regularizing techniques are from:
+ [Regularizing and Optimizing LSTM Language Models](https://arxiv.org/abs/1708.02182)
+ [An Analysis of Neural Language Modeling at Multiple Scales](https://arxiv.org/abs/1803.08240)

Datasets:
+ SEAME is divided into train, valid, test
+ train set is used for adaptation
+ SEAME test for testing, which is further divided into cs, en, zh
+ for replicating INTERSPEECH system
+ OpenSubtitle and others for training the cross-lingual system. Kept consistent with INTERSPEECH paper
+ MORE mono-lingual, and parallel data for training large system (when have time)

Baseline:
+ The baseline model is a transformer model trained on synthetic code-switch data 
+ | end of epoch   5 | time: 5717.78s | valid loss  5.05 | valid ppl   **156.20** |
+ Refer to `cs_baseline.log` or `cs_big_baseline`
+ The baseline system is adapted with real code-switch data as outlined in INTERSPEECH paper
+ Validation perplexity is computed on SEAME test
+ The total vocabulary is 50K, did not trim to 26K because not comparing to other systems, can always trim down to 26k when necessary

Improved system:
+ keeping the same vocabulary, dataset and test set and adaptation set
+ | end of epoch   5 | time: 28817.31s | valid loss  4.90 | valid_en ppl   **134.66** |
+ Prelim result shows that idea is working

Analysis:
+ Generate words from the model (done), perform analysis in term of syntax and code-switching rules
+ Explain in theory or offer intuition about why the attention mechanism work in finding the cross-lingual representation
+ Perform word embedding analysis, in term of BLI, cross-lingual word similarity. Also perform monolingual analysis
+ Visualize attention weights (may not be helpful)

TO-DO:
+ finalize the vocab.count. OK
+ train zero-shot model (without adapt and real cs) to over-fit, about 9 epochs, separately save the best model for testing. OK
+ train with adapt for both baseline and CSLM. left baseline adapt and zero-shot
+ train baseline again, fixed 50k vocab, previous run vocab=55k. OK
+ perform split ppl test
+ extract embedding
+ extract attention weights and plot

Important Model parameters:
```
n_vocab = 50k (keep vocab.count in a central location, always build dictionary from vocab)
max_len = 33 for monolingual, 66 for parallel, 35 and 70 with BOS and EOS tokens (memory concern)
n_ctx = 35 for monolingual, 70 for parallel
emsize : 768
n_heads : 12
nlayers : 12
embd_pdrop : 0.1
dropouti : 0.4
dropouth : 0.15
dropout : 0.4
attn_pdrop : 0.1
resid_pdrop : 0.1
```

Citations:
```
@article{merityRegOpt,
  title={{Regularizing and Optimizing LSTM Language Models}},
  author={Merity, Stephen and Keskar, Nitish Shirish and Socher, Richard},
  journal={arXiv preprint arXiv:1708.02182},
  year={2017}
}
```

```
@article{merityAnalysis,
  title={{An Analysis of Neural Language Modeling at Multiple Scales}},
  author={Merity, Stephen and Keskar, Nitish Shirish and Socher, Richard},
  journal={arXiv preprint arXiv:1803.08240},
  year={2018}
}
```