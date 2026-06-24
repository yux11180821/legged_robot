"""Distributed Hierarchical RL — reproduction of arXiv:2407.06499
("Learning a Distributed Hierarchical Locomotion Controller for Embodied Cooperation").

Method (faithful to the paper):
  * Algorithm  = IPPO (Independent PPO, FULLY DECENTRALIZED) -- NOT MAPPO/CTDE.
                 MAPPO is only kept as the CTDE *baseline* the paper compares against.
  * Policy     = per-agent 3-layer HRL: UL (perception) -> ML (RNN memory) -> LL (frozen operator).
  * Training   = stage 1 single-agent lower operator -> freeze; stage 2 IPPO over UL+ML.
  * Obs        = proprioceptive p_t (lower only) + exteroceptive e_t (env + NEAREST-NEIGHBOR only).

Code is organized in the standard on-policy MARL framework layout
(algorithms / networks / envs / memories / runners / utils / configs).
"""
