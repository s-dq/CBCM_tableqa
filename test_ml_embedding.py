from tqdm import tqdm
import logging
import numpy as np
import ujson as json
from typing import List
import random
from paddlenlp.data import Stack, Tuple, Pad
import glob
import gzip
import bz2
import math
import random
import os
import socket
import time
from dataclasses import dataclass
from typing import Optional, Tuple
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import Tensor
from paddle.nn.layer.layers import in_declarative_mode
from paddlenlp.layers import Linear as TransposedLinear
from paddlenlp.utils.env import CONFIG_NAME
from paddlenlp.transformers import PretrainedModel, register_base_model
from paddlenlp.transformers.model_outputs import (
    BaseModelOutputWithPoolingAndCrossAttentions,
    MaskedLMOutput,
    ModelOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from paddlenlp.transformers.ernie.configuration import (
    ERNIE_PRETRAINED_INIT_CONFIGURATION,
    ERNIE_PRETRAINED_RESOURCE_FILES_MAP,
    ErnieConfig,
)
from paddlenlp.transformers import ErnieTokenizer#,ErnieForSequenceClassification #换成自己改了return的函数



__all__ = [
    "ErnieModel",
    "ErniePretrainedModel",
    "ErnieForSequenceClassification"
]

class ErniePretrainedModel(PretrainedModel):
    model_config_file = CONFIG_NAME
    config_class = ErnieConfig
    resource_files_names = {"model_state": "model_state.pdparams"}
    base_model_prefix = "ernie"

    pretrained_init_configuration = ERNIE_PRETRAINED_INIT_CONFIGURATION
    pretrained_resource_files_map = ERNIE_PRETRAINED_RESOURCE_FILES_MAP

    def _init_weights(self, layer):
        """Initialization hook"""
        if isinstance(layer, (nn.Linear, nn.Embedding)):
            # only support dygraph, use truncated_normal and make it inplace
            # and configurable later
            if isinstance(layer.weight, paddle.Tensor):
                layer.weight.set_value(
                    paddle.tensor.normal(
                        mean=0.0,
                        std=self.config.initializer_range,
                        shape=layer.weight.shape,
                    )
                )
        elif isinstance(layer, nn.LayerNorm):
            layer._epsilon = 1e-12

@register_base_model
class ErnieModel(ErniePretrainedModel):
    def __init__(self, config: ErnieConfig):
        super(ErnieModel, self).__init__(config)
        self.pad_token_id = config.pad_token_id
        self.initializer_range = config.initializer_range
        weight_attr = paddle.ParamAttr(
            initializer=nn.initializer.TruncatedNormal(mean=0.0, std=self.initializer_range)
        )
        self.embeddings = ErnieEmbeddings(config=config, weight_attr=weight_attr)
        encoder_layer = nn.TransformerEncoderLayer(
            config.hidden_size,
            config.num_attention_heads,
            config.intermediate_size,
            dropout=config.hidden_dropout_prob,
            activation=config.hidden_act,
            attn_dropout=config.attention_probs_dropout_prob,
            act_dropout=0,
            weight_attr=weight_attr,
            normalize_before=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, config.num_hidden_layers)
        self.pooler = ErniePooler(config, weight_attr)

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def forward(
            self,
            input_ids: Optional[Tensor] = None,
            token_type_ids: Optional[Tensor] = None,
            position_ids: Optional[Tensor] = None,
            attention_mask: Optional[Tensor] = None,
            task_type_ids: Optional[Tensor] = None,
            past_key_values: Optional[Tuple[Tuple[Tensor]]] = None,
            inputs_embeds: Optional[Tensor] = None,
            use_cache: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time.")

        # init the default bool value
        output_attentions = output_attentions if output_attentions is not None else False
        output_hidden_states = output_hidden_states if output_hidden_states is not None else False
        return_dict = return_dict if return_dict is not None else False
        use_cache = use_cache if use_cache is not None else False
        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]

        if attention_mask is None:
            attention_mask = paddle.unsqueeze(
                (input_ids == self.pad_token_id).astype(self.pooler.dense.weight.dtype) * -1e4, axis=[1, 2]
            )
            if past_key_values is not None:
                batch_size = past_key_values[0][0].shape[0]
                past_mask = paddle.zeros([batch_size, 1, 1, past_key_values_length], dtype=attention_mask.dtype)
                attention_mask = paddle.concat([past_mask, attention_mask], axis=-1)

        # For 2D attention_mask from tokenizer
        elif attention_mask.ndim == 2:
            attention_mask = paddle.unsqueeze(attention_mask, axis=[1, 2]).astype(paddle.get_default_dtype())
            attention_mask = (1.0 - attention_mask) * -1e4

        attention_mask.stop_gradient = True

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            task_type_ids=task_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )

        self.encoder._use_cache = use_cache  # To be consistent with HF
        encoder_outputs = self.encoder(
            embedding_output,
            src_mask=attention_mask,
            cache=past_key_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        if isinstance(encoder_outputs, type(embedding_output)):
            sequence_output = encoder_outputs
            pooled_output = self.pooler(sequence_output)
            return (sequence_output, pooled_output)
        else:
            sequence_output = encoder_outputs[0]
            pooled_output = self.pooler(sequence_output)
            if not return_dict:
                return (sequence_output, pooled_output) + encoder_outputs[1:]
            return BaseModelOutputWithPoolingAndCrossAttentions(
                last_hidden_state=sequence_output,
                pooler_output=pooled_output,
                past_key_values=encoder_outputs.past_key_values,
                hidden_states=encoder_outputs.hidden_states,
                attentions=encoder_outputs.attentions,
            )



class ErnieForSequenceClassification(ErniePretrainedModel):
    def __init__(self, config):
        super(ErnieForSequenceClassification, self).__init__(config)
        self.ernie = ErnieModel(config)
        self.num_labels = config.num_labels
        self.dropout = nn.Dropout(
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(
            self,
            input_ids: Optional[Tensor] = None,
            token_type_ids: Optional[Tensor] = None,
            position_ids: Optional[Tensor] = None,
            attention_mask: Optional[Tensor] = None,
            inputs_embeds: Optional[Tensor] = None,
            labels: Optional[Tensor] = None,
            output_hidden_states: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.ernie(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        pooled_output = outputs[1]

        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!用于分类的tensor
        return outputs[1]
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == paddle.int64 or labels.dtype == paddle.int32):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = paddle.nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = paddle.nn.CrossEntropyLoss()
                loss = loss_fct(logits.reshape((-1, self.num_labels)), labels.reshape((-1,)))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = paddle.nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else (output[0] if len(output) == 1 else output)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class ErnieEmbeddings(nn.Layer):
    def __init__(self, config: ErnieConfig, weight_attr):
        super(ErnieEmbeddings, self).__init__()

        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id, weight_attr=weight_attr
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, weight_attr=weight_attr
        )
        self.type_vocab_size = config.type_vocab_size
        if self.type_vocab_size > 0:
            self.token_type_embeddings = nn.Embedding(
                config.type_vocab_size, config.hidden_size, weight_attr=weight_attr
            )
        self.use_task_id = config.use_task_id
        self.task_id = config.task_id
        if self.use_task_id:
            self.task_type_embeddings = nn.Embedding(
                config.task_type_vocab_size, config.hidden_size, weight_attr=weight_attr
            )
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
            self,
            input_ids: Optional[Tensor] = None,
            token_type_ids: Optional[Tensor] = None,
            position_ids: Optional[Tensor] = None,
            task_type_ids: Optional[Tensor] = None,
            inputs_embeds: Optional[Tensor] = None,
            past_key_values_length: int = 0,
    ):

        if input_ids is not None:
            inputs_embeds = self.word_embeddings(input_ids)

        input_shape = inputs_embeds.shape[:-1] if in_declarative_mode() else paddle.shape(inputs_embeds)[:-1]

        if position_ids is None:
            # maybe need use shape op to unify static graph and dynamic graph
            ones = paddle.ones(input_shape, dtype="int64")
            seq_length = paddle.cumsum(ones, axis=1)
            position_ids = seq_length - ones

            if past_key_values_length > 0:
                position_ids = position_ids + past_key_values_length

            position_ids.stop_gradient = True

        position_embeddings = self.position_embeddings(position_ids)
        embeddings = inputs_embeds + position_embeddings

        if self.type_vocab_size > 0:
            if token_type_ids is None:
                token_type_ids = paddle.zeros(input_shape, dtype="int64")
            token_type_embeddings = self.token_type_embeddings(token_type_ids)
            embeddings = embeddings + token_type_embeddings

        if self.use_task_id:
            if task_type_ids is None:
                task_type_ids = paddle.ones(input_shape, dtype="int64") * self.task_id
            task_type_embeddings = self.task_type_embeddings(task_type_ids)
            embeddings = embeddings + task_type_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class ErniePooler(nn.Layer):
    def __init__(self, config: ErnieConfig, weight_attr):
        super(ErniePooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size, weight_attr=weight_attr)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output





def jsonl_lines(input_files, completed_files=None, limit=0, report_every=100000, *, errors=None, shuffled=None):
    return read_lines(jsonl_files(input_files, completed_files),
                      limit=limit, report_every=report_every,
                      errors=errors, shuffled_files=shuffled)
def jsonl_files(input_files, completed_files=None):
    return [f for f in expand_files(input_files, '*.jsonl*', completed_files) if not f.endswith('.lock')]

def expand_files(input_files, file_pattern='*', completed_files=None):
    if type(input_files) is str:
        if ':' in input_files:
            input_files = input_files.split(':')
        else:
            input_files = [input_files]
    # expand input files recursively
    all_input_files = []
    if completed_files is None:
        completed_files = []
    for input_file in input_files:
        if input_file in completed_files:
            continue
        if not os.path.exists(input_file):
            raise ValueError(f'no such file: {input_file}')
        if os.path.isdir(input_file):
            sub_files = glob.glob(input_file + "/**/" + file_pattern, recursive=True)
            sub_files = [f for f in sub_files if not os.path.isdir(f)]
            sub_files = [f for f in sub_files if f not in input_files and f not in completed_files]
            all_input_files.extend(sub_files)
        else:
            all_input_files.append(input_file)
    all_input_files.sort()
    return all_input_files


def read_open(input_file, *, binary=False, errors=None):
    if binary:
        if input_file.endswith(".gz"):
            return gzip.open(input_file, "rb")
        elif input_file.endswith('.bz2'):
            return bz2.open(input_file, "rb")
        else:
            return open(input_file, "rb")
    else:
        if input_file.endswith(".gz"):
            return gzip.open(input_file, "rt", encoding='utf-8', errors=errors)
        elif input_file.endswith('.bz2'):
            return bz2.open(input_file, "rt", encoding='utf-8', errors=errors)
        else:
            return open(input_file, "r", encoding='utf-8', errors=errors)


def read_lines(input_files, limit=0, report_every=100000, *, errors=None, shuffled_files=None):
    count = 0
    input_files = expand_files(input_files)
    if shuffled_files:
        if type(shuffled_files) != random.Random:
            shuffled_files = random.Random()
        num_open_blocks = int(math.ceil(len(input_files) / 32.0))
        for open_block_i in range(num_open_blocks):
            open_files = [read_open(in_file, errors=errors) for in_file in input_files[open_block_i::num_open_blocks]]
            while len(open_files) > 0:
                fndx = shuffled_files.randint(0, len(open_files) - 1)
                next_line = open_files[fndx].readline()
                if next_line:
                    yield next_line
                    count += 1
                    if count % report_every == 0:
                        logger.info(f'On line {count}')
                else:
                    open_files[fndx].close()
                    del open_files[fndx]
    else:
        for input_file in input_files:
            with read_open(input_file, errors=errors) as reader:
                for line in reader:
                    yield line
                    count += 1
                    if count % report_every == 0:
                        logger.info(f'On line {count} in {input_file}')
                    if 0 < limit <= count:
                        return


def standard_json_mapper(jobj):
    if 'text_b' in jobj and 'question_type' in jobj:
        return jobj['id'], jobj['text_a'], jobj['text_b'], jobj['label'], jobj['question_type']
    elif 'text_b' in jobj and not 'question_type' in jobj:
        return jobj['id'], jobj['text_a'], jobj['text_b'], jobj['label']
    else:
        return jobj['id'], jobj['text'], jobj['label']


class SeqPairInst:
    __slots__ = 'inst_id', 'input_ids', 'token_type_ids', 'label', 'question_type', 'schema'

    def __init__(self, inst_id, input_ids, token_type_ids, label, question_type=None, schema=None):
        self.inst_id = inst_id
        self.input_ids = input_ids  # list
        self.token_type_ids = token_type_ids  # list
        self.label = label
        self.question_type = question_type
        self.schema = schema


class SeqPairDataloader():
    def __init__(self, hypers, per_gpu_batch_size, tokenizer, data_dir,
                 json_mapper=standard_json_mapper, uneven_batches=False):
        super().__init__()
        self.hypers = hypers
        self.per_gpu_batch_size = per_gpu_batch_size
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.cls_id = tokenizer.cls_token_id  # [CLS] 3
        self.sep_id = tokenizer.sep_token_id  # [SEP] 5
        self.pad_id = tokenizer.pad_token_id  # [PAD]  0

        self.json_mapper = json_mapper
        self.uneven_batches = uneven_batches

        self.insts = self.load_data()  #
        self.batch_size = self.per_gpu_batch_size * self.hypers.n_gpu  # 16*1
        self.num_batches = len(self.insts) // self.batch_size
        # self.random=random.Random(123)
        # if self.random is not None:
        #    self.random.shuffle(self.insts)

        if self.uneven_batches or self.hypers.world_size == 1:
            if len(self.insts) % self.batch_size != 0:
                self.num_batches += 1
        logger.info(f'insts size={len(self.insts)}, batch size = {self.batch_size}, batch count = {self.num_batches}')

        self.displayer = self.display_batch

        # just load the entire teacher predictions

    def set_random_to_shuffle_data(self, random_seed):
        self.random = random.Random(random_seed)
        self.random.shuffle(self.insts)

    def __len__(self):
        return self.num_batches

    def __getitem__(self, index):
        if index >= self.num_batches:
            raise IndexError
        if self.hypers.world_size == 1:
            batch_insts = self.insts[index::self.num_batches]
        else:
            batch_insts = self.insts[index * self.batch_size:(index + 1) * self.batch_size]
        batch_tensors = self.make_batch(batch_insts)

        # if index == 0 and self.displayer is not None:
        #    self.displayer(batch_tensors)
        return batch_tensors

    def make_batch(self, insts: List[SeqPairInst]):
        batch_size = len(insts)

        input_id_pad = Pad(axis=0, pad_val=self.tokenizer.pad_token_id)
        token_type_id_pad = Pad(axis=0, pad_val=self.tokenizer.pad_token_type_id)
        label_stack = Stack(dtype="int64")

        ids = [inst.inst_id for inst in insts]
        input_ids_batch = input_id_pad([inst.input_ids for inst in insts])  # [batch,max_length]
        token_type_ids_batch = token_type_id_pad([inst.token_type_ids for inst in insts])  # [batch,max_length]
        labels = label_stack([1 if inst.label else 0 for inst in insts])  # (batch,)

        input_ids_batch = paddle.to_tensor(input_ids_batch)
        token_type_ids_batch = paddle.to_tensor(token_type_ids_batch)
        labels = paddle.to_tensor(labels)
        tensors = ids, input_ids_batch, token_type_ids_batch, labels
        return tensors

    def display_batch(self, batch):

        ids = batch[0]
        input_ids, token_types, labels = [t.cpu().numpy() for t in batch[1:4]]
        print("in dataloader() input_ids.shape:")
        print(input_ids.shape)

        for i in range(1):
            toks = [str for str in self.tokenizer.convert_ids_to_tokens(input_ids[i])]
            logger.info(f"id-:{ids[i]}")
            logger.info(f"tokens:{toks}")
            logger.info(f"token_type_ids:{token_types[i]}")
            logger.info(f"labels:{labels[i]}")

    def load_data(self):
        logger.info('loading data from %s' % (self.data_dir))
        lines = jsonl_lines(self.data_dir)
        insts = []

        for line in lines:
            jobj = json.loads(line)
            # CONSIDER: do multiprocessing?
            question_type = None
            schema = None
            one_item = self.json_mapper(jobj)
            if len(one_item) == 5:
                inst_id, text_a, text_b, label, question_type = one_item
                schema = None
            if len(one_item) == 4:
                inst_id, text_a, text_b, label = one_item
                question_type = None
                schema = None

            multi_seg_input = self.tokenizer(text=text_a, text_pair=text_b, max_seq_len=self.hypers.max_seq_length)

            sp_inst = SeqPairInst(inst_id, multi_seg_input['input_ids'], multi_seg_input['token_type_ids'], label,
                                  question_type, schema)
            insts.append(sp_inst)

        return insts


class HypersBase:
    def __init__(self):
        self.local_rank, self.global_rank, self.world_size = -1, 0, 1
        # required parameters initialized to the datatype
        self.model_type = ''  # 'albert'
        self.model_name_or_path = ''  # albert-base-v2
        self.resume_from = ''  # to resume training from a checkpoint
        self.config_name = ''
        self.tokenizer_name = ''
        self.cache_dir = ''
        self.do_lower_case = False
        self.gradient_accumulation_steps = 1  # 2
        self.learning_rate = 5e-5  # 2e-5
        self.weight_decay = 0.0  # previous default was 0.01  # 0.01
        self.adam_epsilon = 1e-8
        self.max_grad_norm = 1.0
        self.warmup_instances = 0  # previous default was 0.1 of total  # 100000
        self.warmup_fraction = 0.1
        self.num_train_epochs = 3  # 3
        self.no_cuda = False
        self.n_gpu = 1
        self.seed = 42
        self.fp16 = False
        self.fp16_opt_level = 'O1'  # previous default was O2
        self.full_train_batch_size = 8  # previous default was 32    #64
        self.per_gpu_eval_batch_size = 8
        self.output_dir = ''  # where to save model
        self.save_total_limit = 1  # limit to number of checkpoints saved in the output dir
        self.save_steps = 0  # do we save checkpoints every N steps? (TODO: put in terms of hours instead)
        self.use_tensorboard = False
        self.log_on_all_nodes = False
        self.server_ip = ''
        self.server_port = ''
        self.__required_args__ = []  # required args,must specify in the command line

    def set_seed(self, seed=None):
        if seed is None:
            seed = self.seed
        random.seed(seed)
        np.random.seed(seed)
        paddle.seed(seed)

    def set_gradient_accumulation_steps(self):
        if self.n_gpu * self.world_size * self.per_gpu_train_batch_size > self.full_train_batch_size:
            self.per_gpu_train_batch_size = self.full_train_batch_size // (self.n_gpu * self.world_size)
            self.gradient_accumulation_steps = 1
        else:
            self.gradient_accumulation_steps = self.full_train_batch_size // \
                                               (self.n_gpu * self.world_size * self.per_gpu_train_batch_size)

    def _basic_post_init(self):
        # Setup CUDA, GPU

        self.device = paddle.device.get_device()  # "gpu:0" or "cpu"
        if self.n_gpu > 0:
            # 64 /(1*1*4)
            self.per_gpu_train_batch_size = self.full_train_batch_size // \
                                            (self.n_gpu * self.world_size * self.gradient_accumulation_steps)
        else:
            self.per_gpu_train_batch_size = self.full_train_batch_size // self.gradient_accumulation_steps

        self.stop_time = None
        if 'TIME_LIMIT_MINS' in os.environ:
            self.stop_time = time.time() + 60 * (int(os.environ['TIME_LIMIT_MINS']) - 5)

    def _post_init(self):
        self._basic_post_init()

        self._setup_logging()

        logger.warning(
            "On %s, Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
            socket.gethostname(),
            self.local_rank,
            self.device,
            self.n_gpu,
            bool(self.local_rank != -1),
            self.fp16,
        )
        logger.info(f'hypers:\n{self}')

    def _setup_logging(self):
        # force our logging style
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        if self.log_on_all_nodes:
            grank = self.global_rank

            class HostnameFilter(logging.Filter):
                hostname = socket.gethostname()
                if '.' in hostname:
                    hostname = hostname[0:hostname.find('.')]  # the first part of the hostname

                def filter(self, record):
                    record.hostname = HostnameFilter.hostname
                    record.global_rank = grank
                    return True

            handler = logging.StreamHandler()
            handler.setLevel(logging.INFO)
            handler.addFilter(HostnameFilter())
            format = logging.Formatter('%(hostname)s[%(global_rank)d] %(filename)s:%(lineno)d - %(message)s',
                                       datefmt='%m/%d/%Y %H:%M:%S')
            handler.setFormatter(format)
            logging.getLogger('').addHandler(handler)
        else:
            logging.basicConfig(format='%(filename)s:%(lineno)d - %(message)s',
                                datefmt='%m/%d/%Y %H:%M:%S',
                                level=logging.INFO)
        if self.global_rank != 0 and not self.log_on_all_nodes:
            try:
                logging.getLogger().setLevel(logging.WARNING)
            except:
                pass

    def to_dict(self):
        d = self.__dict__.copy()
        del d['device']
        return d

    def from_dict(self, a_dict):
        fill_from_dict(self, a_dict)
        self._basic_post_init()  # setup device and per_gpu_batch_size
        return self

    def __str__(self):
        return json.dumps(self.to_dict(), indent=2)

    def fill_from_args(self):
        fill_from_args(self)
        self._post_init()
        return self

class SeqPairHypers(HypersBase):
    """
    Arguments and hypers
    """
    def __init__(self):
        super().__init__()
        self.max_seq_length = 128
        self.num_labels = 2
        self.single_sequence = False
        self.additional_special_tokens = ''
        self.is_separate = False
        # for reasonable values see the various params.json under
        #    https://github.com/peterliht/knowledge-distillation-pytorch
        self.kd_alpha = 0.9
        self.kd_temperature = 10.0

class SeqPairArgs(SeqPairHypers):
    def __init__(self):
        super().__init__()
        self.train_dir = ''
        self.dev_dir = ''
        self.train_instances = 0  # we need to know the total number of training instances (should just be total line count)
        self.hyper_tune = 0  # number of trials to search hyperparameters
        self.prune_after = 5
        self.save_per_epoch = False
        self.teacher_labels = ''  # the labels from the teacher for the train_dir dataset

args = SeqPairArgs()



# 加载模型
model = ErnieForSequenceClassification.from_pretrained('ernie-1.0-base-zh',num_classes=2)
tokenizer = ErnieTokenizer.from_pretrained('ernie-1.0-base-zh')
tokenizer.add_special_tokens({'additional_special_tokens': ['*']})
test_input = tokenizer("*")
assert test_input['input_ids'][1] != tokenizer.unk_token_id
# 加载训练数据
train_dataloader = SeqPairDataloader(args, 1, tokenizer, './traindata/best/train_cells.jsonl.gz',json_mapper=standard_json_mapper)
# 数据内容：ids,input_ids_batch,token_type_ids_batch,labels=batch
train_data=list(train_dataloader)
# 加载测试数据
test_dataloader = SeqPairDataloader(args, 1, tokenizer, './traindata/best/test_cells.jsonl.gz',json_mapper=standard_json_mapper)
test_data=list(test_dataloader)

# 设置设备为 GPU 或 CPU
device = paddle.set_device("gpu" if paddle.is_compiled_with_cuda() else "cpu")
model = model.to(device)
if not os.path.exists('./traindata/test_ml'):
    os.makedirs('./traindata/test_ml')
# 打开 JSON 文件进行写入
with open('./traindata/test_ml/train.jsonl', 'w') as train_file, open('./traindata/test_ml/test.jsonl', 'w') as test_file:
    # 处理训练数据并逐个保存到 JSON 文件
    for i in tqdm(range(len(train_data)), desc="正在处理训练数据"):
        embedding = model(train_data[i][1], train_data[i][2])[0]
        lable = train_data[i][3]
        id=train_data[i][0]
        embedding = embedding.cpu().numpy().tolist()  # 转换为 numpy 数组再转为 list
        lable = lable.cpu().numpy().tolist()  # 同样处理 y_train
        json.dump({'id':id,'embedding':embedding,'lable':lable}, train_file)
        train_file.write("\n")  # 在每个条目之间添加换行符

    # 处理测试数据并逐个保存到 JSON 文件
    for i in tqdm(range(len(test_data)), desc="正在处理测试数据"):
        embedding = model(test_data[i][1], test_data[i][2])[0]
        lable = test_data[i][3]
        id=test_data[i][0]
        embedding = embedding.cpu().numpy().tolist()  # 转换为 numpy 数组再转为 list
        lable = lable.cpu().numpy().tolist()  # 同样处理 y_test
        json.dump({'id':id,'embedding':embedding,'lable':lable}, test_file)
        test_file.write("\n")  # 在每个条目之间添加换行符


