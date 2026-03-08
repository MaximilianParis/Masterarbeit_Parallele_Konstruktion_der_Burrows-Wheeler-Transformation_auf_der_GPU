Code von Masterabreit.
Benchmark ist auf RTX4000, CUDA 12.8 erstellt worden.
Inputs sind von https://pizzachili.dcc.uchile.cl/repcorpus.html.

Implementiert sind hier die Algorithmen:

@article{8,
  author    = {Boucher, Christina and Gagie, Travis and Kuhnle, Alan and Langmead, Ben and Manzini, Giovanni and Mun, Taher},
  title     = {Prefix-free parsing for building big BWTs},
  journal   = {Algorithms for Molecular Biology},
  year      = {2019},
  volume    = {14},
  number    = {1},
  pages     = {13},
  doi       = {10.1186/s13015-019-0148-5},
  url       = {https://doi.org/10.1186/s13015-019-0148-5},
  issn      = {1748-7188}
}

und


@misc{7,
      title={Fast and memory-efficient BWT construction of repetitive texts using Lyndon grammars}, 
      author={Jannik Olbrich},
      year={2025},
      eprint={2504.19123},
      archivePrefix={arXiv},
      primaryClass={cs.DS},
      url={https://arxiv.org/abs/2504.19123}, 
}.

Der Konkurenalgorithmus ist libcubwt: https://github.com/IlyaGrebnov/libcubwt.
