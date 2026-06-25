"""Task 3 -- Ravine Bridging (Fig.1c, HETEROGENEOUS): one group pushes a movable bridge
so another group can cross; two policies (plane agents vs ravine agents), Eq.2 objective.

This is the third cooperation benchmark (§5.1/Fig.1) and the only HETEROGENEOUS one: the
N agents split into two SPECIES with DISTINCT roles -- "plane" agents that push a movable
bridge into place and "ravine" agents that then cross it.  Because the roles differ, the
two species DO NOT share parameters: each species has its own HRL policy, and the team is
optimised under the multi-species objective of Eq.2 (the heterogeneous generalisation of
the shared-reward Eq.1 used by the homogeneous tasks).  Agents within a species are still
homogeneous (shared policy + shared reward inside the species).  It will likewise be a
concrete ``MultiAgentEnv`` over a ``Backend``, trained in stage 2 (§5.2) on top of the
frozen Lower-Layer locomotion operator.

TODO(code-later): the concrete env (bridge dynamics, two-species observation/reward
wiring, Eq.2 multi-policy training hooks) is not yet implemented; this module is a
placeholder for that task."""
