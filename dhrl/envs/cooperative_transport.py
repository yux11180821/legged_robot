"""Task 1 -- Cooperative Transport (Fig.1a, homogeneous): a group moves a cylinder
object to a target zone collaboratively.  Reward shared across agents.

This is the first of the paper's three cooperation benchmarks (§5.1/Fig.1).  A team of
HOMOGENEOUS agents (one shared HRL policy + one SHARED reward, Eq.1) must collectively
push/carry a cylindrical payload into a goal region -- no single agent can move it alone,
so coordinated pushing must emerge from the decentralized IPPO training over the frozen
Lower-Layer locomotion operator (stage 2 of §5.2).  Like the other tasks it will be
implemented as a concrete ``MultiAgentEnv`` (see ``environment.py``) bound to a
``Backend``, emitting per-agent (e_t, p_t) and the shared cooperative reward each step.

TODO(code-later): the concrete env (object dynamics, goal-zone reward, observation
assembly) is not yet implemented; this module is a placeholder for that task."""
