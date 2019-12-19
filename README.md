# Maximum Entropy
### Density Estimation for Log Determinants and Graph spectra

Brief Explanation
```bash
Given a set of Moments this code calculates the density
of maximum entropy corresponding to those moments using
scipy optimise, gridding and analytical maxent equations.
We show the usecase of this by calculating log determinants
on real datasets and by calculating the similarity between
real roadside, biological and other networks
```

Running the Code

```bash
#MaxEnt distribution saved on a large graph
python3 Amazon_MaxEnt_Chebyshev_shallowP2.py

#Download Amazon and Livejournal datasets from
https://snap.stanford.edu/data/

#Other good graph dataset sources include
http://networkrepository.com/networks.php
https://icon.colorado.edu/#!/networks

#To see the divergence between MaxEnt approximations of the spectra
use the notebook
Measure Graph Divergence/Graph Spectra.ipynb

#Please Cite
@article{granziol2019meme,
  title={MEMe: An Accurate Maximum Entropy Method for Efficient Approximations in Large-Scale Machine Learning},
  author={Granziol, Diego and Ru, Binxin and Zohren, Stefan and Dong, Xiaowen and Osborne, Michael and Roberts, Stephen},
  journal={Entropy},
  volume={21},
  number={6},
  pages={551},
  year={2019},
  publisher={Multidisciplinary Digital Publishing Institute}
}


