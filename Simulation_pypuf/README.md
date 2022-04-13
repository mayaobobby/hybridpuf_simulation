# Simulation with pypuf

There are different files can be found in the project folder. To start with, user can run:
```
python main_template.py
```
to perform a modeling attack on HPUF with an underlying of k-XORPUF (```apuf_attack.py```). It outputs the relation of #CRPs and accuracy of produced models with CPUF/HPUF constructions. 

The different quantum encoding alters the coefficient of HPUF, e.g., the BB84 encoding causes a randomness of 15% on the response. Therefore, the coefficient is 85%. 

With a underlying of k-XORPUF and BB84 encoding, the simulation results is given as follows (with a challenge size n=64/128):
<img alt="alt_text" width="2000px" src="../images/k4.png" />
<img alt="alt_text" width="2000px" src="../images/k5.png" />

Meanwhile, there are more simulations provided in terms of arbitrary distribution of challenge and different instances of CPUFs. 