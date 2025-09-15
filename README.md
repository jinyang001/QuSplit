# QuSplit: Achieving Both High Fidelity and Throughput via Job Splitting on Noisy Quantum Computers

This repository contains the source code for the paper **"QuSplit: Achieving Both High Fidelity and Throughput via Job Splitting on Noisy Quantum Computers"**. It provides the scripts used to generate the figures in the *Observation* section of the paper.  

## Requirements

The project has been tested with the following dependencies:

- numpy==2.0.2  
- matplotlib==3.9.3  
- qiskit==1.2.4  
- qiskit-aer==0.15.1  
- qiskit-algorithms==0.3.1  

You can install them with:

```bash
pip install -r requirements.txt
```

## Usage

To reproduce the figures:
- Figure1
    ```bash
    python observation1and2.py
    ```
- Figure2
    ```bash
    python observation3.py
    ```
  
## Citation
If you find this code useful, please cite our paper:

```bibtex
@article{li2025qusplit,
  title={QuSplit: Achieving Both High Fidelity and Throughput via Job Splitting on Noisy Quantum Computers},
  author={Li, Jinyang and Song, Yuhong and Liu, Yipei and Pan, Jianli and Yang, Lei and Humble, Travis and Jiang, Weiwen},
  journal={arXiv preprint arXiv:2501.12492},
  year={2025}
}
```