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

import random

import torch
import torch.nn.functional as F
from models.base_captioning import BaseCaptioning
from models.ced.audiotransformer import AudioTransformer, CEDConfig
from peft import LoraConfig, TaskType
from transformers import LlamaForCausalLM, LlamaTokenizer


class CedLlama7BCaptionModel(BaseCaptioning):
    def __init__(self, config):
        super().__init__(config)

        # encoder
        ced_config = CEDConfig()

        # the checkpoint can be downloaded from zenodo:
        # https://zenodo.org/record/8275347/files/audiotransformer_base_mAP_4999.pt?download=1
        ced_checkpoint = torch.load(
            "pretrained_models/ced/audiotransformer_base_mAP_4999.pt"
        )
        self.encoder = AudioTransformer(
            ced_config,
            patch_size=16,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            outputdim=527,
            target_length=1012,
        )
        self.encoder.load_state_dict(ced_checkpoint, strict=False)
        encoder_peft_config = LoraConfig(
            target_modules=["q_proj", "v_proj"],
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
        )
        self.apply_encoder_strategy(encoder_peft_config)

        # mlp
        self.speech_former, self.speech_query_tokens = self.build_audio_qformer(
            1, self.encoder.embed_dim, 2, 1
        )

        # decoder
        hf_token = "your huggingface token"
        self.tokenizer = LlamaTokenizer.from_pretrained(
            "meta-llama/Llama-2-7b", token=hf_token
        )
        self.tokenizer.pad_token = self.tokenizer.unk_token
        self.decoder = LlamaForCausalLM.from_pretrained(
            "meta-llama/Llama-2-7b", token=hf_token
        )
        peft_config = LoraConfig(
            target_modules=["q_proj", "v_proj"],
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
        )
        self.apply_decoder_strategy(peft_config)

        # mlp, must call after init self.decoder
        self.enc_to_dec_proj = self.build_audio_projector(
            projector_type="linear", in_dim=self.speech_former.config.hidden_size
        )

    def print_module_parameters(self):
        encoder_num_params = sum([i.numel() for i in self.encoder.parameters()])
        decoder_num_params = sum([i.numel() for i in self.decoder.parameters()])
        speech_former_num_params = sum(
            [i.numel() for i in self.speech_former.parameters()]
        )
        mlp_num_params = sum([i.numel() for i in self.enc_to_dec_proj.parameters()])
        print(
            f"model params encoder: {encoder_num_params}, decoder: {decoder_num_params}, speech_former: {speech_former_num_params}, mlp: {mlp_num_params}"
        )

    def prepare_inputs_labels_for_multimodal(
        self, audio_embeds, atts, prompt, text=None
    ):
        prompt_left = []
        prompt_right = []
        for i, p in enumerate(prompt):
            l, r = p.split("<AcousticTokens>")
            prompt_left.append(self.tokenizer.bos_token + l)
            prompt_right.append(r)

        prompt_left_tokens = self.tokenizer(
            prompt_left, add_special_tokens=False, return_tensors="pt"
        ).to(audio_embeds.device)
        prompt_left_embeds = self.decoder.model.model.embed_tokens(
            prompt_left_tokens.input_ids
        )

        prompt_right_tokens = self.tokenizer(
            prompt_right,
            add_special_tokens=False,
            padding="longest",
            return_tensors="pt",
        ).to(audio_embeds.device)
        prompt_right_embeds = self.decoder.model.model.embed_tokens(
            prompt_right_tokens.input_ids
        )

        input_embeds = torch.cat(
            [prompt_left_embeds, audio_embeds, prompt_right_embeds], dim=1
        )
        input_mask = torch.cat(
            [
                prompt_left_tokens.attention_mask,
                atts,
                prompt_right_tokens.attention_mask,
            ],
            dim=1,
        )

        decoder_targets = None
        if text is not None:
            new_text = []
            for t in text:
                new_text.append(t + self.tokenizer.eos_token)  # </s> is the eos_token
            text_tokens = self.tokenizer(
                new_text,
                add_special_tokens=False,
                padding="longest",
                return_tensors="pt",
            ).to(audio_embeds.device)
            text_embeds = self.decoder.model.model.embed_tokens(text_tokens.input_ids)

            targets = text_tokens.input_ids.masked_fill(
                text_tokens.input_ids == self.tokenizer.pad_token_id, -100
            )
            empty_targets = (
                torch.ones([input_mask.shape[0], input_mask.shape[1]], dtype=torch.long)
                .to(audio_embeds.device)
                .fill_(-100)
            )
            decoder_targets = torch.cat([empty_targets, targets], dim=1)

            input_embeds = torch.cat([input_embeds, text_embeds], dim=1)
            input_mask = torch.cat([input_mask, text_tokens.attention_mask], dim=1)
        return input_embeds, input_mask, decoder_targets

    def forward_encoder(self, audios):
        audio_embeds = self.encoder(audios)

        # Qformer
        batch, tokens, dim = audio_embeds.shape
        kernel = (1, 17)  # for ced 714ms/per frame (ced 10s: 252 frame), we reduce to about 1.4 frames/second
        audio_embeds_new = F.unfold(
            audio_embeds.transpose(1, 2).unsqueeze(2), kernel_size=kernel, stride=kernel
        )
        audio_embeds_new = audio_embeds_new.view(batch, dim, kernel[1], -1)
        audio_embeds_new = torch.permute(audio_embeds_new, [0, 3, 2, 1])
        audio_embeds = audio_embeds_new.reshape(-1, kernel[1], dim)

        speech_atts = torch.ones(
            audio_embeds.size()[:-1], dtype=torch.long, device=audio_embeds.device
        )
        query_tokens = self.speech_query_tokens.expand(audio_embeds.shape[0], -1, -1)
        audio_embeds = self.speech_former.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=audio_embeds,
            encoder_attention_mask=speech_atts,
            return_dict=True,
        )["last_hidden_state"]

        # MLP
        encoder_hidden_states = self.enc_to_dec_proj(audio_embeds)
        encoder_hidden_states = encoder_hidden_states.view(
            batch, -1, encoder_hidden_states.size(2)
        ).contiguous()
        encoder_atts = torch.ones(
            encoder_hidden_states.size()[:-1], dtype=torch.long
        ).to(encoder_hidden_states.device)
        return encoder_hidden_states, encoder_atts

    def forward(self, samples):
        audios = samples["audios"]
        text = samples["text"]

        prompt = [random.choice(self.prompt)] * len(text)
        # encoder
        encoder_hidden_states, encoder_atts = self.forward_encoder(audios)

        input_embeds, input_mask, decoder_targets = (
            self.prepare_inputs_labels_for_multimodal(
                encoder_hidden_states, encoder_atts, prompt, text
            )
        )
        decoder_output = self.decoder(
            input_ids=None,
            inputs_embeds=input_embeds,
            attention_mask=input_mask,
            labels=decoder_targets,
            return_dict=True,
        )
        return decoder_output.loss, decoder_output.logits

    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=3,
        max_length=30,
        min_length=2,
        top_p=0.9,
        repetition_penalty=1.0,
    ):
        audios = samples["audios"].to(self.device)

        prompt = [random.choice(self.prompt)] * audios.shape[0]
        encoder_hidden_states, encoder_atts = self.forward_encoder(audios)
        input_embeds, input_mask, decoder_targets = (
            self.prepare_inputs_labels_for_multimodal(
                encoder_hidden_states, encoder_atts, prompt
            )
        )

        outputs = self.decoder.generate(
            inputs_embeds=input_embeds,
            attention_mask=input_mask,
            max_new_tokens=max_length,
            min_new_tokens=min_length,
            do_sample=use_nucleus_sampling,
            top_p=top_p,
            temperature=1.0,
            num_beams=num_beams,
            repetition_penalty=repetition_penalty,
        )
        captions = self.tokenizer.batch_decode(outputs, add_special_tokens=False)
        return captions
