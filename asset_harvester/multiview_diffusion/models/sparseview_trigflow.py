# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

from .sparseviewdit import SparseViewDiTTransformer2DModelNative


class SparseViewDiTTrigFlow(SparseViewDiTTransformer2DModelNative):
    """TrigFlow wrapper for multiview SparseViewDiT distillation.

    Inherits from SparseViewDiTTransformer2DModelNative so all multiview
    conditioning kwargs (camera_emb, rays, cond_mask, etc.) pass through
    to super().forward() automatically.
    """

    def __init__(self, *args, guidance=False, **kwargs):
        super().__init__(*args, guidance_embeds=guidance, **kwargs)
        self.guidance = guidance
        self.logvar_linear = torch.nn.Linear(self.hidden_size, 1)
        self.do_conversion = True
        torch.nn.init.xavier_uniform_(self.logvar_linear.weight)
        torch.nn.init.constant_(self.logvar_linear.bias, 0)

    def forward(self, hidden_states, timestep, jvp=False, return_logvar=False, **kwargs):
        batch_size = hidden_states.shape[0]
        latents = hidden_states
        # TrigFlow --> Flow Transformation
        if self.do_conversion:
            timestep = timestep.expand(latents.shape[0]).to(hidden_states.dtype)

            flow_timestep = torch.sin(timestep) / (torch.cos(timestep) + torch.sin(timestep))

            flow_timestep_expanded = flow_timestep.view(-1, 1, 1, 1)
            latent_model_input = latents * torch.sqrt(flow_timestep_expanded**2 + (1 - flow_timestep_expanded) ** 2)
            latent_model_input = latent_model_input.to(hidden_states.dtype)
            flow_timestep = flow_timestep * 1000.0
        else:
            flow_timestep = timestep
            latent_model_input = latents
        # forward in original flow — multiview kwargs pass through via **kwargs

        if jvp and self.gradient_checkpointing:
            self.gradient_checkpointing = False
            model_out = super().forward(
                hidden_states=latent_model_input,
                timestep=flow_timestep,
                **kwargs,
            )[0]
            self.gradient_checkpointing = True
        else:
            model_out = super().forward(
                hidden_states=latent_model_input,
                timestep=flow_timestep,
                **kwargs,
            )[0]
        if self.do_conversion:
            # Flow --> TrigFlow Transformation
            trigflow_model_out = (
                (1 - 2 * flow_timestep_expanded) * latent_model_input
                + (1 - 2 * flow_timestep_expanded + 2 * flow_timestep_expanded**2) * model_out
            ) / torch.sqrt(flow_timestep_expanded**2 + (1 - flow_timestep_expanded) ** 2)
        else:
            trigflow_model_out = model_out

        if self.guidance:
            if kwargs.get("guidance", None) is None:
                guidance = torch.zeros(batch_size, device=hidden_states.device, dtype=hidden_states.dtype)
            else:
                guidance = kwargs.get("guidance")
            timestep, embedded_timestep = self.time_embed(timestep, guidance=guidance, hidden_dtype=hidden_states.dtype)
        else:
            timestep, embedded_timestep = self.time_embed(
                timestep, batch_size=batch_size, hidden_dtype=hidden_states.dtype
            )

        if return_logvar:
            logvar = self.logvar_linear(embedded_timestep)
            return trigflow_model_out, logvar

        return (trigflow_model_out,)
