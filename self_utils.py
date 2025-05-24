import os
import pickle
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers import set_seed

from typing import Optional, Tuple, Union, List
from transformers.cache_utils import Cache

from transformers.modeling_outputs import CausalLMOutputWithPast

from transformers import BloomForCausalLM, BloomConfig
from transformers.models.bloom.modeling_bloom import BloomBlock
from transformers import LlamaForCausalLM, LlamaConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers import Qwen2ForCausalLM, Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
from transformers import MistralForCausalLM, MistralConfig
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from torch.utils.tensorboard import SummaryWriter

def sample_gumbel(shape, device, dtype, eps=1e-20):
    u = torch.rand(shape, device=device, dtype=dtype)
    return -torch.log(-torch.log(u+eps) + eps)

def gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, dim=-1):
    gumbels = sample_gumbel(logits.shape, device=logits.device, dtype=logits.dtype)
    gumbels = (logits + gumbels)
    y_soft = gumbels.softmax(dim)

    if hard:
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, device=logits.device, dtype=logits.dtype).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        ret = y_soft
    return ret


class CustomLlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.config = config
        self.layer_idx = layer_idx
        num_classes = config.multi_classes
        lora_alpha = 32
        self.router_logits_linear = nn.Linear(config.hidden_size, num_classes)
        self.Shared_CAM_Trans = nn.Linear(config.hidden_size, config.hidden_size, bias=self.config.use_CAM_Bias)
        self.Multi_LC_Trans = nn.ParameterList([
            nn.ParameterDict({
                'Lora_CAM_A': nn.Linear(config.hidden_size, config.lora_dim, bias=self.config.use_CAM_Bias),
                'Lora_CAM_N': nn.Linear(config.lora_dim, config.lora_dim, bias=self.config.use_CAM_Bias),
                'Lora_CAM_B': nn.Linear(config.lora_dim, config.hidden_size, bias=self.config.use_CAM_Bias),
                'scaling': lora_alpha / config.lora_dim
            }) for _ in range(num_classes)
        ])

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        attn_input = hidden_states.clone()

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        # This will store router logits and gumbel probs if HyCAM is active and training
        router_logits_for_loss = None
        gumbel_probs_for_loss = None

        shared_CAM = F.silu(self.Shared_CAM_Trans(attn_input)) + 1

        bs, token_length, hidden_dim = attn_input.size()
        x_flat = attn_input.view(bs*token_length, hidden_dim)
        router_logits_flat = self.router_logits_linear(x_flat)

        current_routing_probs_flat = F.gumbel_softmax(router_logits_flat, tau=0.5, hard=True)

        if self.training: # Store for load balancing loss calculation
            router_logits_for_loss = router_logits_flat
            gumbel_probs_for_loss = current_routing_probs_flat

        res = [F.silu(
                CAM_Trans['Lora_CAM_B'](CAM_Trans['Lora_CAM_N'](CAM_Trans['Lora_CAM_A'](x_flat)))*CAM_Trans['scaling']) 
                    for CAM_Trans in self.Multi_LC_Trans]
        expert_outputs = torch.stack(res, dim=-1)
        output = torch.sum(expert_outputs * current_routing_probs_flat.unsqueeze(1), dim=-1)
        
        eff_CAM = output.view(bs, token_length, -1)
        CAM = shared_CAM + eff_CAM


        hidden_states = residual + hidden_states

        hidden_states = hidden_states * CAM

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        # Append router logits and gumbel probs if they were computed for loss
        if self.training and router_logits_for_loss is not None and gumbel_probs_for_loss is not None:
            outputs += (router_logits_for_loss, gumbel_probs_for_loss)

        return outputs

class CustomLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.model.layers = nn.ModuleList()
        for i in range(config.num_hidden_layers):
            self.model.layers.append(CustomLlamaDecoderLayer(config, i))
    def forward(
        self,
        input_ids = None,
        attention_mask = None,
        position_ids = None,
        past_key_values = None,
        inputs_embeds = None,
        labels = None,
        use_cache = None,
        output_attentions = None,
        output_hidden_states = None,
        return_dict = None,
        cache_position = None,
        num_logits_to_keep = 0,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        def compute_load_balance_loss(self, router_logits_for_loss, gumbel_probs_for_loss):
            B, N_s = gumbel_probs_for_loss.size()
            softmax_logits = F.softmax(router_logits_for_loss, dim=-1)
            balance_loss = torch.sum(
                (torch.mean(gumbel_probs_for_loss, dim=0) * torch.mean(softmax_logits, dim=0))
            )
            return balance_loss
    
        router_logits_for_loss = outputs[-2]  # Get router logits
        gumbel_probs_for_loss = outputs[-1]   # Get Gumbel softmax probs
        balance_loss = compute_load_balance_loss(router_logits_for_loss, gumbel_probs_for_loss)



        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
            # TODO: remove the float() operation in v4.46
            logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :]).float()

        loss = None
        if labels is not None:
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = logits.float()
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class CustomQwen2DecoderLayer(Qwen2DecoderLayer):
    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.config = config
        self.layer_idx = layer_idx
        num_classes = config.multi_classes
        lora_alpha = 32
        self.router_logits_linear = nn.Linear(config.hidden_size, num_classes)
        self.Shared_CAM_Trans = nn.Linear(config.hidden_size, config.hidden_size, bias=self.config.use_CAM_Bias)
        self.Multi_LC_Trans = nn.ParameterList([
            nn.ParameterDict({
                'Lora_CAM_A': nn.Linear(config.hidden_size, config.lora_dim, bias=self.config.use_CAM_Bias),
                'Lora_CAM_N': nn.Linear(config.lora_dim, config.lora_dim, bias=self.config.use_CAM_Bias),
                'Lora_CAM_B': nn.Linear(config.lora_dim, config.hidden_size, bias=self.config.use_CAM_Bias),
                'scaling': lora_alpha / config.lora_dim
            }) for _ in range(num_classes)
        ])

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        attn_input = hidden_states.clone()

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        # This will store router logits and gumbel probs if HyCAM is active and training
        router_logits_for_loss = None
        gumbel_probs_for_loss = None

        shared_CAM = F.silu(self.Shared_CAM_Trans(attn_input)) + 1

        bs, token_length, hidden_dim = attn_input.size()
        x_flat = attn_input.view(bs*token_length, hidden_dim)
        router_logits_flat = self.router_logits_linear(x_flat)

        if self.training: # Store for load balancing loss calculation
            router_logits_for_loss = router_logits_flat
            gumbel_probs_for_loss = current_routing_probs_flat


        current_routing_probs_flat = F.gumbel_softmax(router_logits_flat, tau=0.5, hard=True)

        res = [F.silu(
                CAM_Trans['Lora_CAM_B'](CAM_Trans['Lora_CAM_N'](CAM_Trans['Lora_CAM_A'](x_flat)))*CAM_Trans['scaling']) 
                    for CAM_Trans in self.Multi_LC_Trans]
        expert_outputs = torch.stack(res, dim=-1)
        output = torch.sum(expert_outputs * current_routing_probs_flat.unsqueeze(1), dim=-1)
        
        eff_CAM = output.view(bs, token_length, -1)
        CAM = shared_CAM + eff_CAM


        hidden_states = residual + hidden_states

        hidden_states = hidden_states * CAM

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        # Append router logits and gumbel probs if they were computed for loss
        if self.training and router_logits_for_loss is not None and gumbel_probs_for_loss is not None:
            outputs += (router_logits_for_loss, gumbel_probs_for_loss)

        return outputs

class CustomQwen2ForCausalLM(Qwen2ForCausalLM):
    def __init__(self, config: Qwen2Config):
        super().__init__(config)
        self.model.layers = nn.ModuleList()
        for i in range(config.num_hidden_layers):
            self.model.layers.append(CustomQwen2DecoderLayer(config, i))

class CustomMistralDecoderLayer(MistralDecoderLayer):
    def __init__(self, config: MistralConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.config = config
        self.layer_idx = layer_idx
        num_classes = config.multi_classes
        lora_alpha = 32
        self.router_logits_linear = nn.Linear(config.hidden_size, num_classes)
        self.Shared_CAM_Trans = nn.Linear(config.hidden_size, config.hidden_size, bias=self.config.use_CAM_Bias)
        self.Multi_LC_Trans = nn.ParameterList([
            nn.ParameterDict({
                'Lora_CAM_A': nn.Linear(config.hidden_size, config.lora_dim, bias=self.config.use_CAM_Bias),
                'Lora_CAM_N': nn.Linear(config.lora_dim, config.lora_dim, bias=self.config.use_CAM_Bias),
                'Lora_CAM_B': nn.Linear(config.lora_dim, config.hidden_size, bias=self.config.use_CAM_Bias),
                'scaling': lora_alpha / config.lora_dim
            }) for _ in range(num_classes)
        ])


    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        attn_input = hidden_states.clone()

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        # This will store router logits and gumbel probs if HyCAM is active and training
        router_logits_for_loss = None
        gumbel_probs_for_loss = None

        shared_CAM = F.silu(self.Shared_CAM_Trans(attn_input)) + 1

        bs, token_length, hidden_dim = attn_input.size()
        x_flat = attn_input.view(bs*token_length, hidden_dim)
        router_logits_flat = self.router_logits_linear(x_flat)

        current_routing_probs_flat = F.gumbel_softmax(router_logits_flat, tau=0.5, hard=True)

        if self.training: # Store for load balancing loss calculation
            router_logits_for_loss = router_logits_flat
            gumbel_probs_for_loss = current_routing_probs_flat


        res = [F.silu(
            CAM_Trans['Lora_CAM_B'](CAM_Trans['Lora_CAM_N'](CAM_Trans['Lora_CAM_A'](x_flat)))*CAM_Trans['scaling']) 
                    for CAM_Trans in self.Multi_LC_Trans]
        expert_outputs = torch.stack(res, dim=-1)
        output = torch.sum(expert_outputs * current_routing_probs_flat.unsqueeze(1), dim=-1)
        
        eff_CAM = output.view(bs, token_length, -1)
        CAM = shared_CAM + eff_CAM


        hidden_states = residual + hidden_states

        hidden_states = hidden_states * CAM

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        # Append router logits and gumbel probs if they were computed for loss
        if self.training and router_logits_for_loss is not None and gumbel_probs_for_loss is not None:
            outputs += (router_logits_for_loss, gumbel_probs_for_loss)

        return outputs

class CustomMistralForCausalLM(MistralForCausalLM):
    def __init__(self, config: MistralConfig):
        super().__init__(config)
        self.model.layers = nn.ModuleList()
        for i in range(config.num_hidden_layers):
            self.model.layers.append(CustomMistralDecoderLayer(config, i))


def load_CAM_model(args, tokenizer, normal=False):
    llm_name = args.model_name_or_path
    if 'llama' in llm_name.lower():
        config = LlamaConfig.from_pretrained(llm_name)
        model_class = CustomLlamaForCausalLM
    elif 'qwen' in llm_name.lower():
        config = Qwen2Config.from_pretrained(llm_name)
        model_class = CustomQwen2ForCausalLM
    elif 'stral' in llm_name.lower():
        config = MistralConfig.from_pretrained(llm_name)
        model_class = CustomMistralForCausalLM
        
    config.multi_classes = args.multi_classes
    config.lora_dim = args.lora_dim
    config.use_cache = False
    if args.no_CAM_Bias:
        config.use_CAM_Bias = False
    else:
        config.use_CAM_Bias = True              
    
    model = model_class.from_pretrained(llm_name, config=config)
    target_params = ['weight', 'bias']
    for layer in model.model.layers:
        for module_name, module in layer.named_modules():
            if module_name in ['CAM_Trans', 'MC_Trans', 'Shared_CAM_Trans', 'Multi_FC_Trans']:
                for param_name, param in module.named_parameters():
                    for target_param in target_params:
                        if target_param in param_name and target_param == 'weight':
                            torch.nn.init.zeros_(param)
                        if target_param in param_name and target_param == 'bias':
                            torch.nn.init.zeros_(param)
            if module_name in ['selector']:
                for param_name, param in module.named_parameters():
                    torch.nn.init.ones_(param)
            if module_name in ['Multi_LC_Trans', 'Shared_LC_Trans']:
                for param_name, param in module.named_parameters():
                    if 'Lora_CAM_A' in param_name and 'weight' in param_name:
                        print(module_name, param_name, 'kaiming_uni')
                        torch.nn.init.kaiming_uniform_(param, a=math.sqrt(5))
                    if 'Lora_CAM_A' in param_name and 'bias' in param_name:
                        print(module_name, param_name, 'zeros_')
                        torch.nn.init.zeros_(param)
                    if 'Lora_CAM_N' in param_name and 'weight' in param_name:
                        print(module_name, param_name, 'kaiming_uni')
                        torch.nn.init.kaiming_uniform_(param, a=math.sqrt(5))
                    if 'Lora_CAM_N' in param_name and 'bias' in param_name:
                        print(module_name, param_name, 'zeros_')
                        torch.nn.init.zeros_(param)
                    if 'Lora_CAM_B' in param_name and 'weight' in param_name:
                        print(module_name, param_name, 'zeros_')
                        torch.nn.init.zeros_(param)
                    if 'Lora_CAM_B' in param_name and 'bias' in param_name:
                        print(module_name, param_name, 'zeros_')
                        torch.nn.init.zeros_(param)
                        
    model.config.end_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id
    model.resize_token_embeddings(int(
        8 *
        math.ceil(len(tokenizer) / 8.0)))  # make the vocab size multiple of 8

    return model

def only_optimize_CAM_parameters(model, force_optimize_params=['CAM_Trans', 'selector', 'MC_Trans', 'Shared_CAM_Trans', 'Multi_LC_Trans', 'layernorm']):
    print('Only_Optim_CAM')
    # turn off the gradient of all the parameters except the LoRA parameters
    for name, param in model.named_parameters():
        param.requires_grad = False
        for target_name in force_optimize_params:
            if target_name in name:
                param.requires_grad = True
                break
    return model