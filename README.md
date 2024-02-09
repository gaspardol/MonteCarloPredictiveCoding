# Monte Carlo Predictive Coding (MCPC): Learning probability distributions of sensory inputs with local synaptic plasticity

## Overview
This repository houses the implementation and experiments supporting the findings presented in our paper, where we propose a novel computational framework called Monte Carlo Predictive Coding (MCPC). Our work is motivated by the hypothesis that the brain uses probabilistic generative models to interpret sensory information optimally. Previous theories in neuroscience, namely predictive coding and neural sampling, have offered partial explanations for how the brain might achieve this. Predictive coding suggests that networks of neurons can learn generative models through local synaptic plasticity, while neural sampling theories demonstrate how stochastic dynamics enable the representation of posterior distributions of the environment's latent states.

MCPC bridges these two theoretical frameworks, presenting a comprehensive model that leverages the strengths of both predictive coding and neural sampling. Our findings show that MCPC-equipped neural networks can learn accurate generative models using local computation and synaptic plasticity mechanisms. These networks are capable of inferring the posterior distributions of latent states based on sensory inputs and can simulate likely sensory inputs in their absence. Importantly, our model also aligns with experimental observations of neural activity variability during perceptual tasks, providing a unified theory of cortical computation that accounts for a broad range of neural data.

## Key Contributions
- **Integration of Predictive Coding and Neural Sampling**: We detail the theoretical underpinnings and practical implementation of combining predictive coding with neural sampling into a coherent framework, MCPC, which enhances our understanding of cortical computation.
- **Local Computation and Plasticity**: Our model demonstrates how precise generative models can be learned through mechanisms that are implementable within the biological constraints of neural networks.
- **Alignment with Experimental Observations**: MCPC not only offers theoretical contributions but also provides a compelling match to experimental data on neural variability, supporting its relevance and applicability to understanding brain function.


## Structure of the repository
The repository includes:
- `figure_2.py`, `figure_3.py`, `figure_4.py`, `figure_5.py`, `supplementary_figure.py`, and `table_1.py` contain the code to recreate the figures and the table from our paper.
- `requirements.txt` contains the python dependencies of this repository.
- `figures//` contains the output figures of the code.
- `models//` contains trained models to generate figures.
- `utils//` contains utility functions
- `predictive_coding//` contains the code to simulate MCPC- and PC- models.
- `Deep_Latent_Gaussian_Models//` contains code to simulate DLGMs.
- `ResNet.py` contains the code to simulate ResNet-9 models.

## Usage
Follow the following steps to clone the code and setup the necessary python libraries:

```bash
git clone https://github.com/gaspardol/MonteCarloPredictiveCoding.git
cd MonteCarloPredictiveCoding
pip install -r requirements.txt
```

To generate the figures of the paper please run

```bash
python figure_2.py
python figure_3.py
python figure_4.py
python figure_5.py
python supplementary_figure.py
python table_1.py
```

## Aknowledgements
This repository builds upon the following repositories/codes:
- https://github.com/YuhangSong/Prospective-Configuration
- https://github.com/yiyuezhuo/Deep-Latent-Gaussian-Models
- https://gist.github.com/atabakd/ed0f7581f8510c8587bc2f41a094b518


## Citation
For those who find our work useful, here is how you can cite it:

```bibtex
@article{Oliviers2024,
  title={Monte Carlo Predictive Coding: A Unifying Theory of Cortical Computation},
  author={Oliviers, Gaspard and Bogacz, Rafal and Meulemans, Alexander},
  journal={Journal Name},
  volume={XX},
  number={YY},
  pages={ZZ--AA},
  year={2024},
  publisher={Publisher}
}
```

## Contact
For any inquiries or questions regarding the project, please feel free to contact Gaspard Oliviers at gaspard.oliviers@pmb.ox.ac.uk.
