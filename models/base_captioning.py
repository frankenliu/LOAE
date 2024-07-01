# Copyright 2023-2024 Xiaomi Corporation and/or its affiliates.
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

import abc
import logging
import re

import torch
from peft import get_peft_model

from models.Qformer import BertConfig, BertLMHeadModel


class BaseCaptioning(abc.ABC, torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cfg = config

        self.encoder = None
        self.decoder = None

        self.prompt = [
            "Describe the detail of this audio:<AcousticTokens>\n---\nDetailed: "
        ]

    @abc.abstractmethod
    def forward_encoder(self, audios):
        """
        forward encoder for audio with Qformer or MLP and so on.
        :param audios:
        :return:
        """
        pass

    @abc.abstractmethod
    def prepare_inputs_labels_for_multimodal(
        self, audio_embeds, atts, prompt, text=None
    ):
        """
        prepare inputs for decoder
        :param audio_embeds: encoder output
        :param atts:
        :param prompt:
        :param text:
        :return:
        """
        print("BaseCaptioning.prepare_inputs_labels_for_multimodal()")
        pass

    @abc.abstractmethod
    def generate(
        self, samples, num_beams=3, max_length=30, min_length=2, repetition_penalty=1.0
    ):
        """
        generate captioning for the audio
        :param samples:
        :param num_beams:
        :param max_length:
        :param min_length:
        :param repetition_penalty:
        :return:
        """
        print("BaseCaptioning.generate()")
        pass

    @abc.abstractmethod
    def print_module_parameters(self):
        pass

    @property
    def device(self):
        return list(self.parameters())[0].device

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def apply_encoder_strategy(self, peft_config=None):
        strategy = self.cfg["encoder_conf"]["encoder_strategy"]
        if strategy == "frozen":
            for p in self.encoder.parameters():
                p.requires_grad = False
            logging.info("freeze encoder done by config.")
        elif strategy == "trainable":
            logging.info("training all encoder parameters.")
        elif strategy == "lora":
            self.encoder = get_peft_model(self.encoder, peft_config)
            logging.info("fine-tuning encoder with lora.")

    def apply_decoder_strategy(self, peft_config=None):
        strategy = self.cfg["decoder_conf"]["decoder_strategy"]
        is_lora = False
        if strategy == "frozen":
            for p in self.decoder.parameters():
                p.requires_grad = False
            print("freeze decoder done by config.")
        elif strategy == "trainable":
            logging.info("training all decoder parameters.")
        elif strategy == "lora":
            self.decoder = get_peft_model(self.decoder, peft_config)
            logging.info("fine-tuning decoder with lora.")
            is_lora = True
        return is_lora

    def build_audio_projector(
        self, projector_type="linear", in_dim=1024
    ):  # example: mlp2x_gelu, mlp3x_gelu
        if projector_type == "linear":
            return torch.nn.Linear(in_dim, self.decoder.config.hidden_size)

        mlp_gelu_match = re.match(r"^mlp(\d+)x_gelu$", projector_type)
        if mlp_gelu_match:
            mlp_depth = int(mlp_gelu_match.group(1))
            modules = [torch.nn.Linear(in_dim, self.decoder.config.hidden_size)]
            for _ in range(1, mlp_depth):
                modules.append(torch.nn.GELU())
                modules.append(
                    torch.nn.Linear(
                        self.decoder.config.hidden_size, self.decoder.config.hidden_size
                    )
                )
            return torch.nn.Sequential(*modules)
        raise ValueError(f"Unknown projector type: {projector_type}")

    def build_audio_qformer(
        self, num_query_token, audio_width, num_hidden_layers=2, cross_attention_freq=1
    ):
        encoder_config = BertConfig()
        encoder_config.num_hidden_layers = num_hidden_layers
        encoder_config.encoder_width = audio_width
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel(config=encoder_config)
        query_tokens = torch.nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        Qformer.cls = None
        Qformer.bert.embeddings.word_embeddings = None
        Qformer.bert.embeddings.position_embeddings = None
        for layer in Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
        return Qformer, query_tokens

    def print_trainable_parameters(self):
        trainable_params = 0
        all_param = 0
        for _, param in self.named_parameters():
            num_params = param.numel()
            # if using DS Zero 3 and the weights are initialized empty
            if num_params == 0 and hasattr(param, "ds_numel"):
                num_params = param.ds_numel

            # Due to the design of 4bit linear layers from bitsandbytes
            # one needs to multiply the number of parameters by 2 to get
            # the correct number of parameters
            if param.__class__.__name__ == "Params4bit":
                num_params = num_params * 2

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params
        print(
            f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {round(100 * trainable_params / all_param, 6)}"
        )

    def shift_tokens_right(
        self, input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int
    ):
        """
        Shift input ids one token to the right.
        """
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
        shifted_input_ids[:, 0] = decoder_start_token_id

        if pad_token_id is None:
            raise ValueError("self.model.config.pad_token_id has to be defined.")
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        return shifted_input_ids
