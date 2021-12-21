# Scaling up MARL in Ray

This is fast MARL baseline built for **ray[rllib]**
> pip install ray==1.8.0 # version is important

Please annotate one line source code to avoid parallel env seed bug
> ray.rllib.evaluation.rollout_worker.py line 508 


### current support algo
- R2D2(IQL)
- VDN
- QMIX
- PG
- A2C
- A3C
- MAA2C
- PPO
- MAPPO
  
### with neural arch
- GRU
- LSTM
- UPDeT

### current support env
- SMAC
- MPE
- Neural-MMO (simple version)

### incorporate more env
- Meta-Drive (have instance in Ray)
- LBF
- RWARE
- MALMO
- Pommerman
- DM Lab
- Derk's Gym
- Hanabi

