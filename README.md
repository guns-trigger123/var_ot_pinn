# The germ of an idea

Inspired by DAS-PINNs [1], we try another generative model instead of KR net [2] (a kind of flow-based model) as used in [1] based on optimal transport, which was proposed in AE-OT [3], to cope with the PDE problem with complex boundaries.

It seems that KR net (maybe most of flow-based models) does not work perfectly well when the support set of the probability distribution to be learned is too complex (i.e. the support set is not simply connected).

# Implementation

We create a pytorch version of KR net. And we try the problem with 2 peaks in DAS-PINNs [1] using both KR net and AE-OT generative model.

# Requirements

pytorch >= 2.1 ( previous versions may work as well )

# Codes reference

DAS-PINNs: https://github.com/MJfadeaway/DAS

AE-OT: https://github.com/k2cu8/pyOMT

# Reference

[1] Tang K, Wan X, Yang C. DAS-PINNs: A deep adaptive sampling method for solving high-dimensional partial differential equations[J]. Journal of Computational Physics, 2023, 476: 111868.

[2] Tang K, Wan X, Liao Q. Adaptive deep density approximation for Fokker-Planck equations[J]. Journal of Computational Physics, 2022, 457: 111080.

[3] An D, Guo Y, Lei N, et al. AE-OT: A new generative model based on extended semi-discrete optimal transport[J]. ICLR 2020, 2019.

# Results of test problems in DAS-PINNs [1]

samples using generative model in AE-OT (green)

![samples using model in AE-OT](https://github.com/guns-trigger123/var_ot_pinn/blob/master/plots/var_1000_2/samples_48.png)

error

![error](https://github.com/guns-trigger123/var_ot_pinn/blob/master/plots/var_1000_2/49_1500.png)
