# Real-Flight Weighted Online Training Design

Date: 2026-04-07

## Summary

This design adds a "real-flight high-impact update" capability to the existing DDPG weight-training pipeline.

The main training body remains AirSim-based pretraining. During later real-flight sessions, online-collected real-flight transitions should influence the model more strongly than their raw sample count would normally allow. The first implementation target is:

- collect real-flight transitions online during flight
- apply model updates at episode end
- make those updates strongly favor real-flight data
- use the updated model for the next real-flight episode

The design must also leave a clean path to a future inflight-update mode, where updates can be applied during a flight rather than only after an episode ends.

## Problem

The current system already supports:

- AirSim virtual training via `multirotor/DDPG_Weight/train_with_airsim_improved.py`
- real-flight online training via `multirotor/DDPG_Weight/train_with_crazyflie_online.py`
- real-flight log training via `multirotor/DDPG_Weight/train_with_crazyflie_logs.py`
- hybrid mirror training via `multirotor/DDPG_Weight/train_with_hybrid.py`

However, it does not provide an explicit mechanism to say:

- "the number of real-flight samples is small"
- "but each real-flight sample should affect the model much more strongly than simulation samples"

Today, the practical workaround is staged training:

1. pretrain in simulation
2. continue training on real flight

That already helps, but it does not make "real-flight influence strength" a configurable system capability.

## Goals

- Keep AirSim pretraining as the main training stage.
- Support online collection of real-flight data during actual flight.
- Add an explicit high-impact update mechanism for real-flight data.
- Make the first release operate in `episode_end` mode for safety.
- Preserve compatibility with the current DDPG environment and server flow.
- Prepare a future extension path to `inflight` update timing.

## Non-Goals

- Replacing Stable-Baselines3 DDPG with a custom RL framework in this phase.
- Reworking `AlgorithmServer` transport, Unity communication, or Crazyflie logging formats.
- Changing the reward definition as part of this feature.
- Making the first release hot-swap model parameters during the same flight.

## Recommended Approach

The recommended implementation is a two-buffer weighted-update design built around the current online DDPG flow.

Core idea:

- keep simulation pretraining unchanged
- during real-flight training, store online real-flight transitions in a dedicated real buffer
- at episode end, run extra update passes that preferentially sample from the real buffer
- optionally mix in a smaller amount of simulation-derived transitions if available, but let real-flight dominate the update batch

This is preferred over a direct inflight-update implementation because it matches the current code structure, reduces flight risk, and still gives real-flight data disproportionately strong influence over the model before the next episode begins.

## Why This Fits the Current Architecture

The current code already has the right separation points:

- training entrypoint: `multirotor/DDPG_Weight/train_with_crazyflie_online.py`
- online environment: `multirotor/DDPG_Weight/envs/crazyflie_weight_env.py`
- simulation pretraining entrypoint: `multirotor/DDPG_Weight/train_with_airsim_improved.py`

The online environment already exposes the full transition loop:

- current observation from real-flight runtime state
- action chosen by the policy
- reward computed after the flight step
- next observation collected from the updated real state

That means the missing capability is not data access. The missing capability is training-time weighting and update scheduling.

## Design Overview

### 1. Update timing modes

Introduce an explicit update timing concept:

- `episode_end`: collect transitions during flight, update the model after the episode, use the new model on the next episode
- `inflight`: reserved for future work; update during flight with stricter safety controls

Phase 1 implements only `episode_end`, but all new interfaces should accept the timing mode so the system is not painted into a corner.

### 2. Real-flight priority trainer

Add a trainer-side component responsible for real-flight weighting behavior.

Suggested responsibility:

- receive transitions collected from the online environment
- store them with a source tag such as `real`
- trigger extra training passes at episode end
- enforce a configurable real-flight influence policy

Suggested name:

- `RealFlightPriorityTrainer`

This component should live under `multirotor/DDPG_Weight/` and remain training-layer only. It should not own flight control.

### 3. Source-aware transition storage

Introduce source-aware storage for transitions.

Minimum design:

- `real_buffer`: stores transitions collected during current and recent real-flight episodes
- optional `sim_buffer`: stores transitions loaded from prior simulation experience if later needed

For the first phase, the feature should work even if only the real buffer is implemented. The pretrained AirSim model parameters already carry simulation knowledge, so phase 1 does not require importing simulation transitions into the online weighted trainer. That is the simplest safe version.

Each transition record should include:

- observation
- action
- reward
- next_observation
- done
- source (`real`)
- episode index
- step index
- collection timestamp

## Weighting Strategy

The first release should support explicit real-flight dominance through trainer configuration.

Recommended controls:

- `real_update_multiplier`
  - how many extra gradient-update rounds are run from real-flight data after each episode
- `real_batch_ratio`
  - target fraction of real-flight samples inside a mixed batch
- `min_real_samples_before_update`
  - avoids unstable updates from tiny sample counts
- `max_real_updates_per_episode`
  - caps compute time and reduces overfitting risk
- `real_buffer_capacity`
  - how much recent real-flight experience is retained

Recommended phase-1 default behavior:

- use only real-flight transitions for the post-episode high-impact update
- if there are too few real samples, skip the weighted update for that episode

This keeps the implementation simple and makes the meaning of "real-flight high weight" direct and auditable.

## Data Flow

### AirSim pretraining stage

1. Train in AirSim using the existing virtual pipeline.
2. Save the pretrained DDPG model.
3. Enter real-flight training using that model as the initialization point.

### Real-flight episode stage

1. `train_with_crazyflie_online.py` loads the pretrained model.
2. `CrazyflieOnlineWeightEnv` runs the flight episode normally.
3. After each environment step, the transition is copied into the real buffer.
4. During the flight, control continues using the stable model parameters from episode start.
5. At episode end, the priority trainer runs the configured real-flight weighted updates.
6. If update validation passes, the updated model becomes the control model for the next episode.

## Component Boundaries

### `train_with_crazyflie_online.py`

Responsibilities after the change:

- parse new weighting-related config
- construct the priority trainer
- load the pretrained model
- orchestrate the sequence:
  - fly episode
  - collect real transitions
  - run post-episode weighted update
  - checkpoint model and training metadata

It should remain the orchestration entrypoint rather than absorbing buffer or weighting logic directly.

### `CrazyflieOnlineWeightEnv`

Responsibilities after the change:

- continue producing standard Gym-style transitions
- expose enough per-step data for the trainer to store transitions cleanly
- avoid owning weighting policy or update scheduling

The environment should stay focused on interaction with the real platform.

### New trainer helper

Responsibilities after the change:

- store source-aware transitions
- execute post-episode weighted updates
- return update summary metrics
- support future extension to `inflight`

## Configuration Design

Extend the online training config with a nested block such as:

```json
{
  "crazyflie_online": {
    "update_timing": "episode_end",
    "enable_real_weighting": true,
    "real_update_multiplier": 4,
    "real_batch_ratio": 1.0,
    "min_real_samples_before_update": 32,
    "max_real_updates_per_episode": 8,
    "real_buffer_capacity": 5000,
    "rollback_on_bad_update": true
  }
}
```

Interpretation:

- `real_batch_ratio = 1.0` means post-episode weighted updates use only real-flight data
- later, mixed-batch strategies can use values like `0.7`
- `rollback_on_bad_update` enables safety rollback if update validation fails

## Safety and Failure Handling

Safety is the main reason to ship `episode_end` first.

Required protections:

- no parameter hot-swap during the same flight in phase 1
- skip update if the episode produced too few valid transitions
- keep a pre-update checkpoint before weighted update begins
- rollback to the pre-update checkpoint if:
  - loss becomes non-finite
  - update crashes
  - post-update sanity checks fail

Recommended sanity checks:

- no NaN/Inf in policy parameters
- parameter delta below a configurable bound
- sampled action remains inside expected range on a small fixed probe-state set

## Observability

The system should log enough information to explain how strongly real-flight data affected the model.

Required logging:

- episode real sample count
- whether weighted update ran or was skipped
- number of extra update rounds
- effective batch composition
- parameter delta summary
- rollback events

Recommended output fields:

- `real_samples_collected`
- `weighted_update_rounds`
- `weighted_update_status`
- `real_buffer_size`
- `policy_param_delta_norm`
- `rollback_triggered`

## Testing Strategy

### Unit-level

- transition storage correctness
- weighting config parsing
- update skip conditions
- rollback path

### Integration-level

- load pretrained AirSim model into online trainer
- run a mocked real-flight episode
- confirm transitions enter the real buffer
- confirm episode-end weighted update is triggered
- confirm updated model is used on the next episode

### Safety regression

- verify no update is applied mid-episode in `episode_end` mode
- verify failures restore the pre-update model

## Migration Path to Future `inflight` Mode

The first implementation should intentionally leave these extension points:

- `update_timing` enum-like config
- trainer API that can be called either at step time or episode end
- model handoff mechanism separated from environment stepping

When moving to `inflight`, the main changes should be:

- add step-interval or time-interval update triggers
- add stricter action-delta and parameter-delta guards
- optionally use a shadow-model promotion flow instead of direct hot-swap

If these boundaries are respected now, moving from `episode_end` to `inflight` later should be a moderate extension rather than a redesign.

## Risks

- Real-flight sample count may be too small in a single episode to support stable updates.
- Over-weighting real data may overfit to one flight condition or one battery state.
- Tight coupling to SB3 internals could make upgrades harder if implementation reaches too deeply into library-private APIs.

## Decision

Implement phase 1 as:

- AirSim pretraining unchanged
- real-flight online collection unchanged at the environment boundary
- new source-aware real buffer
- explicit post-episode weighted updates with real-flight dominance
- configuration prepared for future `inflight` mode

This gives the system a true "real-flight high influence" capability without introducing same-flight hot-update risk in the first release.
