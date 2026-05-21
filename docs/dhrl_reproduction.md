# D-HRL Reproduction Notes

This project now includes code for the paper `2407.06499v1`, not only the single-RL PPO
baseline used earlier for PointNav curves.

## Paper-to-Code Mapping

| Paper component | Code |
| --- | --- |
| Upper Layer (exteroceptive feature extraction) | `dhrl.models.DistributedHierarchicalActorCritic.upper` |
| Middle Layer (RNN spatiotemporal memory) | `dhrl.models.DistributedHierarchicalActorCritic.middle` |
| Lower Layer (pretrained locomotion controller interface) | `dhrl.models.LowerLayerCommandAdapter`, `dhrl.isaac_adapter` |
| Distributed scalable partial observation | `dhrl.proxy_envs.BaseProxyEnv._nearest_rel` |
| IPPO multi-agent optimization | `dhrl.ippo.train` |
| No Hierarchy ablation | `dhrl.models.NoHierarchyActorCritic` |
| No Spatiotemporal Memory ablation | `dhrl.models.NoMemoryActorCritic` |
| Cooperative Transport | `dhrl.proxy_envs.CooperativeTransportEnv` |
| Corridor Crossing | `dhrl.proxy_envs.CorridorCrossingEnv` |
| Ravine Bridging | `dhrl.proxy_envs.RavineBridgingEnv` |

## What Is Still Needed For Paper-Exact Numbers

The paper's reported success rates require:

1. IsaacSim/Gym assets for the three environments.
2. A pretrained Ant lower-layer locomotion controller.
3. The exact physical reset distributions and reward coefficients.

The current proxy tasks are intentionally lightweight so the algorithm and ablations can be run
without Isaac. They should be treated as an executable reproduction scaffold, not as the final
Isaac benchmark result.
