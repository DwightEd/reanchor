import numpy as np
from decoding.structural_export import cross_head_spread


def test_head_spread_excludes_between_layer_mean_difference():
    x=np.array([[[0.,0.],[.2,.4]],[[1.,1.],[.6,.8]]])
    np.testing.assert_allclose(cross_head_spread(x),[0.,.1],atol=1e-14)
    assert x.std((0,2))[0]>.4
