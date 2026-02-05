from __future__ import annotations

import torch

from persona_steering_library.rl.gspo import (
    RewardWeights,
    approx_sequence_kl,
    combined_reward,
    gspo_sequence_loss,
    z_normalize,
)


def test_z_normalize_constant_values():
    out = z_normalize([5.0, 5.0, 5.0])
    assert out == [0.0, 0.0, 0.0]


def test_gspo_sequence_loss_expected_value():
    logp_old = torch.zeros(2)
    logp_cur = torch.log(torch.tensor([1.1, 0.9]))
    advantages = torch.ones(2)
    loss = gspo_sequence_loss(
        logp_cur=logp_cur, logp_old=logp_old, advantages=advantages, clip_epsilon=0.2
    )
    assert torch.isclose(loss, torch.tensor(-1.0), atol=1e-6)


def test_approx_sequence_kl():
    cur = torch.tensor([2.0, 4.0])
    ref = torch.tensor([1.0, 1.0])
    out = approx_sequence_kl(logp_cur=cur, logp_ref=ref)
    assert torch.isclose(out, torch.tensor(2.0))


def test_combined_reward_with_covert_penalty():
    weights = RewardWeights(alignment=1.0, semantic=0.5, fluency=0.2, covert=0.3)
    components = {
        "alignment": 0.8,
        "semantic": 0.4,
        "fluency_nll": 1.5,
        "covert_detect_p": 0.5,
    }
    reward = combined_reward(components, weights)
    expected = 0.8 - 0.5 * (1.0 - 0.4) - 0.2 * 1.5 - 0.3 * 0.5
    assert abs(reward - expected) < 1e-6
