# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from typing import List, Optional, Tuple, Union
import torch
from torch import nn
from transformers.cache_utils import Cache
from transformers.utils import logging as transfologging
from transformers.models.llama.modeling_llama import LlamaAttention, repeat_kv, apply_rotary_pos_emb
from transformers.models.deberta_v2.modeling_deberta_v2 import DisentangledSelfAttention
from transformers import __version__ as transformers_version





logger = transfologging.get_logger(__name__)

if transformers_version == "4.47.1":
    class LlamaUnmaskedAttention(LlamaAttention):
        def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Cache] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            cache_position: Optional[torch.LongTensor] = None,
            position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
            **kwargs,
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
            
            """
            Modified method to export unscaled cross attention scores along with attention_weights
            """

            bsz, q_len, _ = hidden_states.size()

            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

            # use -1 to infer num_heads and num_key_value_heads as they may vary if tensor parallel is used
            query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

            if position_embeddings is None:
                logger.warning_once(
                    "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                    "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                    "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                    "removed and `position_embeddings` will be mandatory."
                )
                cos, sin = self.rotary_emb(value_states, position_ids)
            else:
                cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

            if past_key_value is not None:
                # sin and cos are specific to RoPE models; cache_position needed for the static cache
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

            unmasked_attn = attn_weights
            if attention_mask is not None:  # no matter the length, we just slice it
                causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
                attn_weights = attn_weights + causal_mask

            # upcast attention to fp32
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
            attn_output = torch.matmul(attn_weights, value_states)

            if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
                raise ValueError(
                    f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                    f" {attn_output.size()}"
                )

            attn_output = attn_output.transpose(1, 2).contiguous()

            attn_output = attn_output.reshape(bsz, q_len, -1)

            attn_output = self.o_proj(attn_output)

            if not output_attentions:
                resu = (attn_output, None, past_key_value)

            else:
                resu = (attn_output, (attn_weights, unmasked_attn), past_key_value)

            return resu
        
elif transformers_version == "4.48.2":
    def llama_custom_attn_function(
        module: nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        scaling: float,
        dropout: float = 0.0,
        **kwargs,
    ):
        key_states = repeat_kv(key, module.num_key_value_groups)
        value_states = repeat_kv(value, module.num_key_value_groups)

        attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
        unmasked_attn = attn_weights
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()

        return attn_output, (attn_weights, unmasked_attn)
    

    def modernBert_custom_attn(
        module, #: "ModernBertAttention"
        qkv: torch.Tensor,
        attention_mask: torch.Tensor,
        sliding_window_mask: torch.Tensor,
        position_ids: Optional[torch.LongTensor],
        local_attention: Tuple[int, int],
        bs: int,
        dim: int,
        output_attentions: Optional[bool] = False,
        **_kwargs,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        # qkv: [batch_size, seqlen, 3, nheads, headdim]
        cos, sin = module.rotary_emb(qkv, position_ids=position_ids)
        query, key, value = qkv.transpose(3, 1).unbind(dim=2)
        # query, key, value: [batch_size, heads, seq_len, head_dim]
        query, key = apply_rotary_pos_emb(query, key, cos, sin)

        scale = module.head_dim**-0.5
        attn_weights = torch.matmul(query, key.transpose(2, 3)) * scale
        unmasked_attn = attn_weights

        if local_attention != (-1, -1):
            attention_mask = sliding_window_mask

        attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=module.attention_dropout, training=module.training)
        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bs, -1, dim)
        if output_attentions:
            return (attn_output, (attn_weights, unmasked_attn))
        return (attn_output,)
    

@torch.jit.script
def scaled_size_sqrt(query_layer: torch.Tensor, scale_factor: int):
    return torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)


class ModifiedDisentangledSelfAttention(DisentangledSelfAttention):
    def forward(
        self,
        hidden_states,
        attention_mask,
        output_attentions=False,
        query_states=None,
        relative_pos=None,
        rel_embeddings=None,
    ):
        """
        Modified method to export unscaled cross attention scores along with attention_probs
        """
        if query_states is None:
            query_states = hidden_states
        query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)

        rel_att = None
        # Take the dot product between "query" and "key" to get the raw attention scores.
        scale_factor = 1
        if "c2p" in self.pos_att_type:
            scale_factor += 1
        if "p2c" in self.pos_att_type:
            scale_factor += 1
        scale = scaled_size_sqrt(query_layer, scale_factor)
        unmasked_attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        if self.relative_attention:
            rel_embeddings = self.pos_dropout(rel_embeddings)
            rel_att = self.disentangled_attention_bias(
                query_layer, key_layer, relative_pos, rel_embeddings, scale_factor
            )

        if rel_att is not None:
            unmasked_attention_scores = unmasked_attention_scores + rel_att
        unmasked_attention_scores = unmasked_attention_scores
        unmasked_attention_scores = unmasked_attention_scores.view(
            -1, self.num_attention_heads, unmasked_attention_scores.size(-2), unmasked_attention_scores.size(-1)
        )

        attention_mask = attention_mask.bool()
        masked_attention_scores = unmasked_attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        # bsz x height x length x dimension
        attention_probs = nn.functional.softmax(masked_attention_scores, dim=-1)
        attention_probs.masked_fill(attention_mask, 0)

        attention_probs = self.dropout(attention_probs)
        context_layer = torch.bmm(
            attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
        )
        context_layer = (
            context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        new_context_layer_shape = context_layer.size()[:-2] + (-1,)
        context_layer = context_layer.view(new_context_layer_shape)
        if not output_attentions:
            return (context_layer, None)
        return (context_layer, (attention_probs, unmasked_attention_scores))
    


