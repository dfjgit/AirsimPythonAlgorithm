# Real Flight Weighted Training

This note summarizes the current weighted online training support for the
Crazyflie real-flight pipeline.

## Current Capabilities

- Optional weighted online updates driven by real-flight transitions.
- Episode-boundary updates only. The runner invokes `model.learn(...)` one
  episode at a time so SB3 still completes its normal rollout storage and
  baseline training before each weighted pass.
- Safety rollback support when weighted updates fail sanity checks.
- Backward compatible: if the `real_weighting` block is missing, the legacy
  `model.learn(...)` path is used.

## Config Keys

`real_weighting` (object, optional):

- `update_timing`: when to apply weighted updates. Currently supported:
  `episode_end` only.
- `enable_real_weighting`: toggle weighted updates. Defaults to `true` when the
  block exists.
- `real_update_multiplier`: multiplier for extra gradient steps per episode.
- `min_real_samples_before_update`: minimum real samples required to trigger a
  weighted update.
- `max_real_updates_per_episode`: clamp on extra gradient steps per episode.
- `real_buffer_capacity`: capacity of the real-flight transition store.
- `rollback_on_bad_update`: rollback policy weights when update sanity checks
  fail.

## Reserved Key

- `real_batch_ratio`: reserved for a future release. The current implementation
  accepts the key for backward compatibility but does not change sampling
  behavior with it.

## Minimal Run Command

```
python train_with_crazyflie_online.py --config configs/crazyflie_online_train_config.json
```
