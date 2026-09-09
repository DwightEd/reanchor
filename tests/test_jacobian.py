import torch

from reanchor.tracing.jacobian import rms_jvp, rms_vjp
from reanchor.tracing.propagation import route_hops


def test_rms_jvp_matches_autograd_and_its_adjoint():
    torch.manual_seed(7)
    reference = torch.randn(5, 12, dtype=torch.float64)
    delta = torch.randn_like(reference)
    reader = torch.randn_like(reference)
    weight = torch.rand(12, dtype=torch.float64)
    epsilon = 1e-5

    def rms_norm(change):
        value = reference + change
        return value * torch.rsqrt(value.square().mean(-1, keepdim=True) + epsilon) * weight

    _, expected = torch.autograd.functional.jvp(rms_norm, torch.zeros_like(reference), delta)
    actual = rms_jvp(delta, reference, weight, epsilon)

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        (actual * reader).sum(),
        (delta * rms_vjp(reader, reference, weight, epsilon)).sum(),
    )


def test_route_hops_keeps_zero_one_and_two_plus_partitions_exact():
    state = torch.tensor([1.0, 10.0, 100.0])
    same = torch.tensor([2.0, 20.0, 200.0])
    cross = torch.tensor([3.0, 30.0, 300.0])

    result = route_hops(state, same, cross)

    torch.testing.assert_close(result, torch.tensor([3.0, 33.0, 630.0]))
