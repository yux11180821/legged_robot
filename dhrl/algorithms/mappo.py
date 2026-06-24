"""MAPPO — CTDE BASELINE ONLY (the method the paper compares against and beats, Fig.5 b/c).

Same actor as IPPO but with a CENTRALIZED critic over the joint observation of all
agents.  Kept solely to reproduce the scalability comparison ("Distributed HRL (IPPO)
vs Centralized Training (MAPPO)"); it is NOT the proposed method.
TODO(code-later).
"""
