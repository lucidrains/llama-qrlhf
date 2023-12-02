## Llama - QRLHF (wip)

Implementation of the Llama (or any language model) architecture with RLHF + Q-learning.

This is experimental / independent open research, built off nothing but speculation. But I'll throw some of my brain cycles at the problem in the coming month, just in case the rumors have any basis. Anything you PhD students can get working is up for grabs.

Will start off by adapting the autoregressive discrete Q-learning formulation in the cited paper below and run a few experiments on arithmetic, using a symbolic solver as reward generator.

<a href="https://www.youtube.com/watch?v=nOBm4aYEYR4">Yannic Kilcher's educational Q-learning video</a>

## Citations

```bibtex
@inproceedings{qtransformer,
    title   = {Q-Transformer: Scalable Offline Reinforcement Learning via Autoregressive Q-Functions},
    authors = {Yevgen Chebotar and Quan Vuong and Alex Irpan and Karol Hausman and Fei Xia and Yao Lu and Aviral Kumar and Tianhe Yu and Alexander Herzog and Karl Pertsch and Keerthana Gopalakrishnan and Julian Ibarz and Ofir Nachum and Sumedh Sontakke and Grecia Salazar and Huong T Tran and Jodilyn Peralta and Clayton Tan and Deeksha Manjunath and Jaspiar Singht and Brianna Zitkovich and Tomas Jackson and Kanishka Rao and Chelsea Finn and Sergey Levine},
    booktitle = {7th Annual Conference on Robot Learning},
    year   = {2023}
}
```
