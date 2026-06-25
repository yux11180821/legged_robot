"""MAPPO -- CTDE BASELINE ONLY (the method the paper compares against and beats, Fig.5 b/c).

This module is the placeholder for the centralized-training, decentralized-execution (CTDE)
baseline used in the paper's scalability comparison "Distributed HRL (IPPO) vs Centralized
Training (MAPPO)".  The intended implementation reuses the SAME shared HRL actor as IPPO (the
trainable UL+ML over the frozen LL, §4.2/§4.3) but pairs it with the ``CentralizedCritic`` from
``critic.py`` -- a value function over the JOINT observation of all agents instead of each
agent's own obs.  That centralized critic input grows with the agent count, which is exactly why
MAPPO fails to converge / transfer as the swarm grows and why the paper's decentralized IPPO
beats it (§5.4, Fig.5 b/c).

It is kept SOLELY to reproduce that comparison and is NOT the proposed method; the decentralized
IPPO in ``ippo.py`` is.  No classes or functions are defined yet -- the concrete trainer is still
to be written.
TODO(code-later).
"""
