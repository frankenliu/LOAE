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

import logging
import math
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import torch
# temp fix bug https://stackoverflow.com/questions/76911396/the-error-of-torch-compile-with-the-cuda12-1
import torch._dynamo
import torch.distributed as dist
import transformers
import yaml
from torch.utils.data import Dataset
from transformers import (HfArgumentParser, Trainer, TrainingArguments,
                          set_seed, trainer_pt_utils, trainer_utils)

from dataset import AudioDataset, collate_fn
from models.loae import CedLlama7BCaptionModel
from utils.utils import get_cpu_mem_info, get_gpu_info, get_tcp_address

torch._dynamo.config.suppress_errors = True


class LoaeTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(inputs)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        return (loss, outputs) if return_outputs else loss

    def log(self, logs: Dict[str, float]) -> None:
        log_str_list = []
        for key, value in logs.items():
            log_str_list.append("{} {}".format(key, round(value, 12)))
        logging.info(
            "epoch {}/{}, step {}, {}".format(
                round(self.state.epoch, 3),
                self.state.num_train_epochs,
                self.state.global_step,
                ",".join(log_str_list),
            )
        )

        # print cpu and gpu info
        if self.state.global_step % 1000 == 0:
            gpu_id = int(os.getenv("CUDA_VISIBLE_DEVICES"))
            gpu_info = get_gpu_info(gpu_id)
            cpu_mem = get_cpu_mem_info()
            logging.info(
                "step {} gpu {} info: mem({}/{}/{})MB, rate:{}% cpu mem:{}/{:.3f}/{:.3f} GB".format(
                    self.state.global_step,
                    gpu_id,
                    gpu_info[0],
                    gpu_info[1],
                    gpu_info[2],
                    gpu_info[3],
                    cpu_mem["count"],
                    cpu_mem["virt_mem"],
                    cpu_mem["res_mem"],
                )
            )

    def evaluate(
        self,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if isinstance(eval_dataset, dict):
            metrics = {}
            for eval_name, _eval_dataset in eval_dataset.items():
                dataset_metrics = self.evaluate(
                    _eval_dataset,
                    ignore_keys=ignore_keys,
                    metric_key_prefix=f"{metric_key_prefix}_{eval_name}",
                )
                metrics.update(dataset_metrics)
            return metrics

        self._memory_tracker.start()
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        start_time = time.time()

        model = self._wrap_model(self.model, training=False, dataloader=eval_dataloader)
        num_examples = self.num_examples(eval_dataloader)
        model.eval()

        total_loss = 0.0
        total_samples = 0
        with torch.no_grad():
            for i, batch_data in enumerate(eval_dataloader):
                loss = self.compute_loss(model, batch_data)
                if torch.isfinite(loss):
                    batch_size = trainer_pt_utils.find_batch_size(batch_data)
                    total_samples += batch_size
                    total_loss += loss.item() * batch_size

        eval_time = time.time() - start_time
        avg_loss = total_loss / (total_samples if total_samples > 0 else 1)
        metrics = {"{}_loss".format(metric_key_prefix): avg_loss, "time": eval_time}
        # save eval loss for each Evaluation Dataset
        self.state.log_history.append({"{}_loss".format(metric_key_prefix): avg_loss})

        total_batch_size = self.args.eval_batch_size * self.args.world_size
        metrics.update(
            trainer_utils.speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=num_examples,
                num_steps=math.ceil(num_examples / total_batch_size),
            )
        )
        logging.info(
            "{} for epoch {}, samples {}/{}, steps {}, loss {}".format(
                metric_key_prefix,
                self.state.epoch,
                total_samples,
                num_examples,
                self.state.global_step,
                avg_loss,
            )
        )

        self._memory_tracker.stop_and_update_metrics(metrics)
        return metrics


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    config_path: Optional[str] = field(default=None, metadata={"help": "setting files"})
    out_dir: Optional[str] = field(
        default=None, metadata={"help": "output dir for model"}
    )
    data_dir: Optional[str] = field(
        default=None, metadata={"help": "train and dev data file in dir"}
    )
    resume_checkpoint: Optional[str] = field(
        default="none", metadata={"help": "resume the model from checkpoint"}
    )
    rank: Optional[int] = field(
        default=0, metadata={"help": "the rank for distributed training"}
    )
    world_size: Optional[int] = field(
        default=1, metadata={"help": "the total gpu number for distributed training"}
    )
    init_model_path: Optional[str] = field(
        default="none", metadata={"help": "init the model weight by other model"}
    )

    def __post_init__(self):
        if self.config_path is None:
            raise ValueError("config path should not none")


def main():
    parser = HfArgumentParser(DataTrainingArguments)
    data_args = parser.parse_args_into_dataclasses()[0]
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    transformers.logging.set_verbosity_info()
    logging.info(data_args)

    gpu_id = int(os.getenv("CUDA_VISIBLE_DEVICES"))
    data_args.rank = data_args.rank - 1
    logging.info("using gpu id:{}".format(gpu_id))

    # load config
    set_seed(20)
    with open(data_args.config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    ddp_dir = os.path.join(data_args.out_dir, "ddp")
    os.makedirs(ddp_dir, exist_ok=True)
    tcp_address = get_tcp_address(ddp_dir, data_args.rank, data_args.world_size)
    logging.info("tcp address:{}".format(tcp_address))

    if data_args.world_size > 1:
        os.environ["LOCAL_RANK"] = "0"
        init_method = "tcp://{}".format(tcp_address)
        logging.info("ddp init method:{}".format(init_method))
        dist.init_process_group(
            "nccl",
            init_method=init_method,
            world_size=data_args.world_size,
            rank=data_args.rank,
        )
        logging.info(
            "torch.distributed is initialized with backend=nccl, init method=%s, world-size=%d, rank=%d, "
            "gpu id=%d" % (init_method, data_args.world_size, data_args.rank, gpu_id)
        )

    dataset_batch_size = config["dataset_conf"]["batch_size"]
    num_workers = config["dataset_conf"]["num_workers"]
    num_epoch = config["epochs"]
    training_args = TrainingArguments(
        output_dir=data_args.out_dir,
        seed=20,
        do_train=True,
        do_eval=True,
        dataloader_num_workers=num_workers,
        remove_unused_columns=False,
        torch_compile=True,
        save_strategy="epoch",
        save_total_limit=num_epoch,
        greater_is_better=False,
        metric_for_best_model="eval_clo_loss",
        load_best_model_at_end=False,
        per_device_train_batch_size=dataset_batch_size,
        evaluation_strategy="epoch",
        per_device_eval_batch_size=dataset_batch_size,
        save_safetensors=False,
        logging_dir=os.path.join(data_args.out_dir, "log"),
        max_grad_norm=config["clip_grad"],
        gradient_accumulation_steps=config["acc_grad"],
    )
    training_args = training_args.set_optimizer(
        config["optim_args"]["name"],
        learning_rate=config["optim_args"]["lr"],
        weight_decay=config["optim_args"]["weight_decay"],
    )
    training_args = training_args.set_lr_scheduler(
        "cosine", num_epoch, warmup_ratio=config["warmup_radio"]
    )
    training_args = training_args.set_logging(
        strategy="steps", steps=100, report_to=[], level="info", first_step=True
    )
    logging.info(training_args)

    train_data_file = os.path.join(
        os.path.join(data_args.data_dir, "train"), "format.data"
    )
    ac_val_data_file = os.path.join(
        os.path.join(data_args.data_dir, "val/ac_val"), "format.data"
    )
    clo_val_data_file = os.path.join(
        os.path.join(data_args.data_dir, "val/clo_val"), "format.data"
    )

    sample_rate, max_length = (
        config["dataset_conf"]["sample_rate"],
        config["dataset_conf"]["max_len"],
    )
    train_dataset = AudioDataset(
        train_data_file, sample_rate=sample_rate, max_length=max_length
    )
    ac_val_dataset = AudioDataset(
        ac_val_data_file, sample_rate=sample_rate, max_length=max_length
    )
    clo_val_dataset = AudioDataset(
        clo_val_data_file, sample_rate=sample_rate, max_length=max_length
    )
    eval_dataset = {"clo": clo_val_dataset, "ac": ac_val_dataset}

    model = CedLlama7BCaptionModel(config)
    if data_args.init_model_path != "none":
        init_state = torch.load(data_args.init_model_path, map_location="cpu")
        model.load_state_dict(init_state)
        # release memory
        del init_state
        logging.info("Loaded init weight from {}".format(data_args.init_model_path))

    logging.info(model)
    model.print_trainable_parameters()
    model.print_module_parameters()

    trainer = LoaeTrainer(
        model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
        compute_metrics=None,
    )

    checkpoint = None
    if data_args.resume_checkpoint != "none":
        checkpoint = data_args.resume_checkpoint

    trainer.train(resume_from_checkpoint=checkpoint)
    logging.info("Training done.")


if __name__ == "__main__":
    main()
