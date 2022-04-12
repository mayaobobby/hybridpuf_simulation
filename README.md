# HybridPUF Simulation

This project is to simulate the hybrid PUF performance with **pypuf** (https://github.com/nils-wisiol/pypuf) and **netsquid** (https://netsquid.org) in python envitonment. More infomation of Hybrid PUF can refer to: https://arxiv.org/abs/2110.09469

## Simualtion with pypuf
Firstly, we only simulate the performance with hybrid PUF encoding technique compared to the classical structure. More simulation on quantum communication is with netsquid.

### Prerequsite

* pypuf

### Deployment
Run ```main_template.py``` with command:
```
python main_template.py
```
Change this file to simulate of different scenarios. 

## Simualtion with netsquid

An easy demo of 2-party Hybrid PUF-based Authentication Protocol (BB84 states) with Lockdown Technique is shown in ```./template.py```. (TBA: Loss/noise simulations and adversarial model)

