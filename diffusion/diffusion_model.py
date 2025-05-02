"""
This code started out as a PyTorch port of Ho et al's diffusion models:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py

Docstrings have been added, as well as DDIM sampling and a new collection of beta schedules.
"""
# MIT License
# Copyright (c) 2021 OpenAI
#
# This code is based on https://github.com/GuyTevet/motion-diffusion-model
# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import torch
import torch as th

from diffusion.gaussian_diffusion import (
    GaussianDiffusion,
    LossType,
    ModelMeanType,
    ModelVarType,
)
from diffusion.multi_scale_init import(
    humanml_init_s1_to_s2,
    humanml_init_s1_to_s3,
)


class DiffusionModel(GaussianDiffusion):
    def __init__(
        self,
        **kwargs,
    ):
        super(DiffusionModel, self).__init__(
            **kwargs,
        )

    def masked_l2(self, a, b):
        bs, n, c = a.shape
#        print('target的形状是',a.shape)  

        loss = torch.mean(
            torch.norm(
                (a - b).reshape(-1, 6),
                2,
                1,
            )
        )

        return loss

    def training_losses(
        self, model, x_start, t, sparse, model_kwargs=None, noise=None, dataset=None
    ):

        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise)

        terms = {}

        if self.loss_type == LossType.KL or self.loss_type == LossType.RESCALED_KL:
            terms["loss"] = self._vb_terms_bpd(
                model=model,
                x_start=x_start,
                x_t=x_t,
                t=t,
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )["output"]
            if self.loss_type == LossType.RESCALED_KL:
                terms["loss"] *= self.num_timesteps
        elif self.loss_type == LossType.MSE or self.loss_type == LossType.RESCALED_MSE:
            model_outputs3,model_outputs2,model_output = model(x_t, self._scale_timesteps(t), sparse, **model_kwargs)

            if self.model_var_type in [
                ModelVarType.LEARNED,
                ModelVarType.LEARNED_RANGE,
            ]:
                B, C = x_t.shape[:2]
                assert model_output.shape == (B, C * 2, *x_t.shape[2:])
                model_output, model_var_values = th.split(model_output, C, dim=1)
                # Learn the variance using the variational bound, but don't let
                # it affect our mean prediction.
                frozen_out = th.cat([model_output.detach(), model_var_values], dim=1)
                terms["vb"] = self._vb_terms_bpd(
                    model=lambda *args, r=frozen_out: r,
                    x_start=x_start,
                    x_t=x_t,
                    t=t,
                    clip_denoised=False,
                )["output"]
                if self.loss_type == LossType.RESCALED_MSE:
                    # Divide by 1000 for equivalence with initial implementation.
                    # Without a factor of 1/1000, the VB term hurts the MSE term.
                    terms["vb"] *= self.num_timesteps / 1000.0

            target = {
                ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                    x_start=x_start, x_t=x_t, t=t
                )[0],
                ModelMeanType.START_X: x_start,
                ModelMeanType.EPSILON: noise,
            }[self.model_mean_type]

            assert model_output.shape == target.shape == x_start.shape
            target_s2=humanml_init_s1_to_s2()(x_start)
            target_s3=humanml_init_s1_to_s3()(x_start)
            back_s2=humanml_init_s1_to_s2()(model_output)
            back_s3=humanml_init_s1_to_s3()(model_output)
            loss1=self.masked_l2(target_s3,model_outputs3)
            loss2=self.masked_l2(target_s2,model_outputs2)
            loss_final=self.masked_l2(target,model_output)
            loss_backs3=self.masked_l2(back_s3,model_outputs3)
            loss_backs2=self.masked_l2(back_s2,model_outputs2)#alignment loss
            terms["rot_mse"] = 0.2*loss1+0.3*loss2+loss_final  #0.2 and 0.3 can be changed for your need, just customized weights
            '''
            terms["rot_mse"] = self.masked_l2(
                target,                                         ###256*196*132=batchsize*seq_len*features
                model_output,
            )
            '''
            terms["loss"] = terms["rot_mse"] + terms.get("vb", 0.0)

        else:
            raise NotImplementedError(self.loss_type)

        return terms
