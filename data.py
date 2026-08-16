import copy
import random
import argparse
import json
import os
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import json
import numpy as np
from collections import defaultdict


def resolve_index_path(data_path, dataset, index_file):
    index_path = Path(index_file)
    if index_path.is_absolute() or index_path.parent != Path("."):
        return index_path
    return Path(data_path) / f"{dataset}{index_file}"


class BaseDataset(Dataset):

    def __init__(self, args):
        super().__init__()

        self.args = args
        self.dataset = args.dataset
        self.data_path = os.path.join(args.data_path, self.dataset)

        self.max_his_len = args.max_his_len
        self.his_sep = args.his_sep
        self.index_file = args.index_file
        self.text_index_file = args.text_index_file
        self.image_index_file = args.image_index_file
        self.fusion_source_modal_mix_prob = getattr(args, "fusion_source_modal_mix_prob", 0.0)
        self.add_prompt = args.add_prompt
        self.modal_mix_prob = getattr(args, "modal_mix_prob", 0.0)

        self.new_tokens = None
        self.allowed_tokens = None
        self.all_items = None

        self.image_indices = None
        self.text_indices = None

        self.fg_image_indices = None


    def _load_data(self):

        with open(os.path.join(self.data_path, self.dataset + ".inter.json"), 'r') as f:
            self.inters = json.load(f)

        with open(resolve_index_path(self.data_path, self.dataset, self.index_file), 'r') as f:
            self.indices = json.load(f)
        with open(resolve_index_path(self.data_path, self.dataset, self.image_index_file), 'r') as f:
            image_indices = json.load(f)

        self.image_indices = {}
        for k, v in image_indices.items():
            if random.random() > 0.5:
                self.image_indices[k] = v

    def get_new_tokens(self):

        if self.new_tokens is not None:
            return self.new_tokens

        self.new_tokens = set()
        for index in self.indices.values():
            for token in index:
                self.new_tokens.add(token)

        if self.image_indices is not None:
            for index in self.image_indices.values():
                for token in index:
                    self.new_tokens.add(token)

        if self.text_indices is not None:
            for index in self.text_indices.values():
                for token in index:
                    self.new_tokens.add(token)

        if self.fg_image_indices is not None:
            for index in self.fg_image_indices.values():
                for token in index:
                    self.new_tokens.add(token)

        self.new_tokens = sorted(list(self.new_tokens))

        return self.new_tokens

    def get_all_tokens(self):

        if self.new_tokens is not None:
            return self.new_tokens

        if self.args.tasks == 'seqrec':
            prefix_list = ["<a_{}>","<b_{}>","<c_{}>","<d_{}>"]

        elif self.args.tasks == 'seqimage':
            prefix_list = ["<A_{}>","<B_{}>","<C_{}>","<D_{}>"]

        else:
            prefix_list = ["<a_{}>","<b_{}>","<c_{}>","<d_{}>", "<A_{}>","<B_{}>","<C_{}>","<D_{}>"]

        new_tokens = set()

        for prefix in prefix_list:
            for i in range(self.args.code_num):
                token = prefix.format(i)
                new_tokens.add(token)

        self.new_tokens = sorted(list(new_tokens))

        # if 'image' in self.args.

        return self.new_tokens

    def get_all_items(self):

        if self.all_items is not None:
            return self.all_items

        self.all_items = set()
        for index in self.indices.values():
            self.all_items.add("".join(index))

        return self.all_items

    def get_prefix_allowed_tokens_fn(self, tokenizer):


        if self.allowed_tokens is None:
            self.allowed_tokens = {}
            for index in self.indices.values():
                for i, token in enumerate(index):
                    token_id = tokenizer(token)["input_ids"][1]
                    if i not in self.allowed_tokens.keys():
                        self.allowed_tokens[i] = set()
                    self.allowed_tokens[i].add(token_id)
            self.allowed_tokens[len(self.allowed_tokens.keys())] = set([tokenizer.eos_token_id])
        sep = [0]


        def prefix_allowed_tokens_fn(batch_id, sentence):
            sentence = sentence.tolist()
            reversed_sent = sentence[::-1]
            for i in range(len(reversed_sent)):
                if reversed_sent[i:i + len(sep)] == sep[::-1]:
                    # print(list(self.allowed_tokens[i]))
                    return list(self.allowed_tokens[i])

        return prefix_allowed_tokens_fn

    def get_prefix_allowed_tokens_fn_new(self, tokenizer):


        if self.allowed_tokens is None:
            self.allowed_tokens = defaultdict(set)
            first_ids = set()
            end_ids = set()
            for index in self.indices.values():
                token_id_list = [tokenizer(token)["input_ids"][1] for token in index]
                for i, token_id in enumerate(token_id_list):
                    if i == 0:
                        first_ids.add(token_id)
                    elif i == len(token_id_list) - 1:
                        end_ids.add(token_id)

                    if i < len(token_id_list) - 1:
                        self.allowed_tokens[token_id].add(token_id_list[i+1])

            for ids in end_ids:
                self.allowed_tokens[ids] = first_ids

        def prefix_allowed_tokens_fn(batch_id, sentence):
            sentence = sentence.tolist()

            return list(self.allowed_tokens[sentence[-1]])

        return prefix_allowed_tokens_fn

    def _process_data(self):

        raise NotImplementedError



class SeqRecDataset(BaseDataset):

    def __init__(self, args, task='seqrec', mode="train",
                 prompt_sample_num=1, prompt_id=0, sample_num=-1):
        super().__init__(args)

        self.mode = mode
        self.prompt_id = prompt_id
        self.sample_num = sample_num
        self.eval_start = getattr(args, "eval_start", 0)
        self.eval_count = getattr(args, "eval_count", -1)
        self.eval_stride = getattr(args, "eval_stride", 1)
        self.eval_indices_file = getattr(args, "eval_indices_file", "")
        self.train_data_mode = args.train_data_mode
        self.add_prompt = args.add_prompt
        self.task = task
        # self.soft_prompt = args.soft_prompts[task]

        # load data
        self._load_data()
        self._remap_items()

        # load data
        if self.mode == 'train':
            self.inter_data = self._process_train_data()
        elif self.mode == 'valid':
            self.inter_data = self._process_valid_data()
        elif self.mode == 'test':
            self.inter_data = self._process_test_data()
        else:
            raise NotImplementedError



    def _load_data(self):

        with open(os.path.join(self.data_path, self.dataset + ".inter.json"), 'r') as f:
            self.inters = json.load(f)
        if self.task == 'seqrec':
            with open(resolve_index_path(self.data_path, self.dataset, self.index_file), 'r') as f:
                self.indices = json.load(f)
        elif self.task == 'seqtext':
            with open(resolve_index_path(self.data_path, self.dataset, self.text_index_file), 'r') as f:
                self.indices = json.load(f)
        elif self.task == 'seqimage':
            with open(resolve_index_path(self.data_path, self.dataset, self.args.image_index_file), 'r') as f:
                self.indices = json.load(f)

        self.modal_mix_indices = None
        self.modal_mix_lookup = None
        if self.mode == 'train' and self.task == 'seqrec' and self.modal_mix_prob > 0:
            modal_mix_indices = {}
            paired_path = resolve_index_path(self.data_path, self.dataset, self.index_file)
            text_path = resolve_index_path(self.data_path, self.dataset, self.text_index_file)
            image_path = resolve_index_path(self.data_path, self.dataset, self.args.image_index_file)

            if os.path.exists(paired_path):
                with open(paired_path, 'r') as f:
                    modal_mix_indices["paired"] = json.load(f)
            if os.path.exists(text_path):
                with open(text_path, 'r') as f:
                    modal_mix_indices["text"] = json.load(f)
            if os.path.exists(image_path):
                with open(image_path, 'r') as f:
                    modal_mix_indices["image"] = json.load(f)

            if len(modal_mix_indices) > 1:
                self.modal_mix_indices = modal_mix_indices
                self.modal_mix_lookup = {
                    "".join(tokens): item_id
                    for item_id, tokens in modal_mix_indices["paired"].items()
                }

    def _remap_items(self):

        self.remapped_inters = dict()
        for uid, items in self.inters.items():
            new_items = ["".join(self.indices[str(i)]) for i in items]
            self.remapped_inters[uid] = new_items


    def _process_train_data(self):

        inter_data = []

        if self.train_data_mode == 0: ## origin, for encoder-decoder
            for uid  in self.remapped_inters:
                items = self.remapped_inters[uid][:-2]
                for i in range(1, len(items)):
                    one_data = dict()
                    # one_data["user"] = uid
                    one_data["item"] = items[i]
                    history = items[:i]
                    if self.max_his_len > 0:
                        history = history[-self.max_his_len:]

                    one_data["inters"] = history
                    inter_data.append(one_data)

        elif self.train_data_mode == 1:  ## all max length
            for uid  in self.remapped_inters:
                items = self.remapped_inters[uid][:-2]

                if len(items) <= self.max_his_len + 1:
                    one_data = dict()
                    one_data["item"] = items[-1]
                    history = items[:-1]
                    one_data["inters"] = history
                    inter_data.append(one_data)
                else:
                    for i in range(self.max_his_len + 1, len(items)):
                        one_data = dict()
                        one_data["item"] = items[i]
                        history = items[:i]
                        history = history[-self.max_his_len:]
                        one_data["inters"] = history
                        inter_data.append(one_data)
        else:
            for uid  in self.remapped_inters:
                items = self.remapped_inters[uid][:-2]
                inter_data.append(items)

        return inter_data

    def _sample_inter_data(self, inter_data):
        if self.sample_num > 0 and self.sample_num < len(inter_data):
            all_inter_idx = range(len(inter_data))
            sample_idx = np.random.choice(all_inter_idx, self.sample_num, replace=False)
            inter_data = np.array(inter_data)[sample_idx].tolist()
        eval_indices_file = getattr(self, "eval_indices_file", "")
        if eval_indices_file:
            indices = self._load_eval_indices(eval_indices_file, len(inter_data))
            inter_data = [inter_data[index] for index in indices]
        eval_start = getattr(self, "eval_start", 0)
        eval_count = getattr(self, "eval_count", -1)
        eval_stride = getattr(self, "eval_stride", 1)
        if eval_start < 0:
            raise ValueError("--eval_start must be non-negative")
        if eval_count < -1:
            raise ValueError("--eval_count must be -1 or non-negative")
        if eval_stride <= 0:
            raise ValueError("--eval_stride must be positive")
        if eval_stride > 1:
            inter_data = inter_data[::eval_stride]
        if eval_start > 0 or eval_count >= 0:
            end = None if eval_count < 0 else eval_start + eval_count
            inter_data = inter_data[eval_start:end]
        return inter_data

    @staticmethod
    def _load_eval_indices(path, row_count):
        with open(path, "r") as f:
            text = f.read().strip()
        if not text:
            return []
        if text[0] == "[":
            indices = json.loads(text)
            if not isinstance(indices, list):
                raise ValueError("--eval_indices_file JSON content must be a list")
        else:
            indices = [int(line.strip()) for line in text.splitlines() if line.strip()]
        clean_indices = []
        seen_indices = set()
        for index in indices:
            index = int(index)
            if index < 0 or index >= row_count:
                raise ValueError(
                    f"eval index {index} out of range for {row_count} rows"
                )
            if index in seen_indices:
                raise ValueError(f"duplicate eval index {index}")
            seen_indices.add(index)
            clean_indices.append(index)
        return clean_indices

    def _process_valid_data(self):

        inter_data = []
        for row_idx, uid in enumerate(self.remapped_inters):
            items = self.remapped_inters[uid]
            one_data = dict()
            one_data["user"] = row_idx
            one_data["item"] = items[-2]
            history = items[:-2]
            if self.max_his_len > 0:
                history = history[-self.max_his_len:]

            one_data["inters"] = history
            inter_data.append(one_data)

        return self._sample_inter_data(inter_data)

    def _process_test_data(self):

        inter_data = []
        for row_idx, uid in enumerate(self.remapped_inters):
            items = self.remapped_inters[uid]
            one_data = dict()
            one_data["user"] = row_idx
            one_data["item"] = items[-1]
            history = items[:-1]
            if self.max_his_len > 0:
                history = history[-self.max_his_len:]

            one_data["inters"] = history
            inter_data.append(one_data)

        return self._sample_inter_data(inter_data)

    def set_prompt(self, prompt_id):

        self.prompt_id = prompt_id

    def __len__(self):

        return len(self.inter_data)

    def _get_text_data(self, data, prompt):

        instruction = prompt["instruction"].format(**data)
        response = prompt["response"].format(**data)

        # input = sft_prompt.format(instruction = instruction, response = "")
        # output = sft_prompt.format(instruction = instruction, response = response)

        # if self.mode == 'test':
        #     return input, response

        return instruction, response

    def __getitem__(self, index):

        if self.mode != 'train' or self.train_data_mode <= 1:
            d = self.inter_data[index]

        elif self.train_data_mode == 2:
            d = dict()
            items = self.inter_data[index]

            if len(items) <= self.max_his_len + 1:
                d["item"] = items[-1]
                d["inters"] = items[:-1]
            else:
                end = random.randint(self.max_his_len + 1, len(items) - 1)
                start = end - self.max_his_len
                d["item"] = items[end]
                d["inters"] = items[start : end]

        elif self.train_data_mode == 3:
            d = dict()
            items = self.inter_data[index]

            end = random.randint(1, len(items) - 1)
            d["item"] = items[end]
            history = items[:end]
            history = history[-self.max_his_len:]
            d["inters"] = history

        else:
            d = dict()
            items = self.inter_data[index]

            start = random.randint(0, len(items) - 2)
            end = random.randint(start + 1, len(items) - 1)

            d["item"] = items[end]
            history = items[start:end]
            history = history[-self.max_his_len:]
            d["inters"] = history

        if self.add_prompt:
            if self.mode == 'train':
                prompt_id = random.randint(0, len(self.prompts) - 1)
            else:
                prompt_id = self.prompt_id

            prompt = self.prompts[prompt_id]

            data = {}
            data["item"] = d["item"]
            history = d["inters"]
            if self.mode == 'train' and self.task == 'seqrec':
                history = self._mix_history_modalities(history)
            data["inters"] = self.his_sep.join(history)

            input, output = self._get_text_data(data, prompt)

        else:
            history = d["inters"]
            if self.mode == 'train' and self.task == 'seqrec':
                history = self._mix_history_modalities(history)
            input = ''.join(history)
            output = d["item"]

        return dict(input_ids=input, labels=output, label=d.get("user", index))

    def _mix_history_modalities(self, history):

        if not self.modal_mix_indices or self.modal_mix_prob <= 0:
            return history

        mixed_history = []
        modal_keys = list(self.modal_mix_indices.keys())
        for item_tokens in history:
            if random.random() >= self.modal_mix_prob:
                mixed_history.append(item_tokens)
                continue

            matched_item_id = self.modal_mix_lookup.get(item_tokens)
            if matched_item_id is None:
                mixed_history.append(item_tokens)
                continue

            candidate_modalities = [
                key for key in modal_keys
                if key != "paired" and matched_item_id in self.modal_mix_indices[key]
            ]
            if not candidate_modalities:
                mixed_history.append(item_tokens)
                continue

            sampled_modality = random.choice(candidate_modalities)
            mixed_history.append("".join(self.modal_mix_indices[sampled_modality][matched_item_id]))

        return mixed_history


class HybridSeqRecDataset(BaseDataset):

    def __init__(self, args, predict_mode='item', mode="train",
                 prompt_sample_num=1, prompt_id=0, sample_num=-1):

        super().__init__(args)

        self.mode = mode
        self.predict_mode = predict_mode
        self.prompt_sample_num = prompt_sample_num
        self.prompt_id = prompt_id
        self.sample_num = sample_num


        self.soft_prompt = args.soft_prompts.get('hybrid', "")


        self._load_data()


        if self.mode == 'train':
            self.inter_data = self._process_train_data()
        elif self.mode == 'valid':
            self.inter_data = self._process_valid_data()
        elif self.mode == 'test':
            self.inter_data = self._process_test_data()
        else:
            raise NotImplementedError

    def _load_data(self):

        with open(os.path.join(self.data_path, self.dataset + ".inter.json"), 'r') as f:
            self.inters = json.load(f)

        with open(resolve_index_path(self.data_path, self.dataset, self.index_file), 'r') as f:
            self.indices = json.load(f)

        with open(resolve_index_path(self.data_path, self.dataset, self.args.image_index_file), 'r') as f:
            self.image_indices = json.load(f)

    def _build_hybrid_history(self, item_list):

        hybrid_history = []
        for item_id in item_list:

            id_tokens = "".join(self.indices[str(item_id)])

            img_tokens = "".join(self.image_indices[str(item_id)])


            hybrid_history.append(id_tokens + img_tokens)

        return hybrid_history

    def _process_train_data(self):
        inter_data = []
        for uid in self.inters:
            items = self.inters[uid][:-2]


            for i in range(1, len(items)):
                one_data = dict()
                target_item = items[i]
                history_items = items[:i]


                if self.max_his_len > 0:
                    history_items = history_items[-self.max_his_len:]



                hybrid_hist_list = self._build_hybrid_history(history_items)
                one_data["inters"] = "".join(hybrid_hist_list)


                if self.predict_mode == 'item':
                    one_data["label"] = "".join(self.indices[str(target_item)])
                elif self.predict_mode == 'image':
                    one_data["label"] = "".join(self.image_indices[str(target_item)])
                else:
                    raise ValueError("predict_mode must be 'item' or 'image'")

                inter_data.append(one_data)
        return inter_data

    def _process_valid_data(self):
        inter_data = []
        for uid in self.inters:
            items = self.inters[uid]
            target_item = items[-2]
            history_items = items[:-2]

            if self.max_his_len > 0:
                history_items = history_items[-self.max_his_len:]

            one_data = dict()
            hybrid_hist_list = self._build_hybrid_history(history_items)
            one_data["inters"] = "".join(hybrid_hist_list)

            if self.predict_mode == 'item':
                one_data["label"] = "".join(self.indices[str(target_item)])
            else:
                one_data["label"] = "".join(self.image_indices[str(target_item)])

            inter_data.append(one_data)
        return inter_data

    def _process_test_data(self):
        inter_data = []
        for uid in self.inters:
            items = self.inters[uid]
            target_item = items[-1]
            history_items = items[:-1]

            if self.max_his_len > 0:
                history_items = history_items[-self.max_his_len:]

            one_data = dict()
            hybrid_hist_list = self._build_hybrid_history(history_items)
            one_data["inters"] = "".join(hybrid_hist_list)

            if self.predict_mode == 'item':
                one_data["label"] = "".join(self.indices[str(target_item)])
            else:
                one_data["label"] = "".join(self.image_indices[str(target_item)])

            inter_data.append(one_data)


        if self.sample_num > 0:
            import numpy as np
            all_inter_idx = range(len(inter_data))
            sample_idx = np.random.choice(all_inter_idx, self.sample_num, replace=False)
            inter_data = np.array(inter_data)[sample_idx].tolist()

        return inter_data

    def __len__(self):
        return len(self.inter_data)

    def __getitem__(self, index):
        d = self.inter_data[index]


        input_seq = self.soft_prompt + d['inters']
        output_seq = d['label']

        return dict(input_ids=input_seq, labels=output_seq)

class ItemImageDataset(BaseDataset):

    def __init__(self, args, task="item2image", prompt_sample_num=1, sample_num=-1):
        super().__init__(args)

        self.task = task.lower()
        self.prompt_sample_num = prompt_sample_num
        self.sample_num = sample_num

        self.soft_prompt = args.soft_prompts[self.task]

        self.text_index_file = args.text_index_file
        self.image_index_file = args.image_index_file


        self._load_data()
        self.data_pair = self._process_data()


    def _load_data(self):

        with open(resolve_index_path(self.data_path, self.dataset, self.index_file), 'r') as f:
            self.indices = json.load(f)
        with open(resolve_index_path(self.data_path, self.dataset, self.text_index_file), 'r') as f:
            self.text_indices = json.load(f)
        with open(resolve_index_path(self.data_path, self.dataset, self.image_index_file), 'r') as f:
            self.image_indices = json.load(f)

    def _resolve_modal_tokens(self, modality, item_id):
        if modality == 'item':
            return "".join(self.indices[item_id])
        if modality == 'text':
            return "".join(self.text_indices[item_id])
        if modality == 'image':
            return "".join(self.image_indices[item_id])
        raise ValueError(f"Unsupported modality: {modality}")

    def _resolve_pair_modalities(self):
        task_to_modalities = {
            'item2image': ('item', 'image'),
            'image2item': ('image', 'item'),
            'item2text': ('item', 'text'),
            'text2item': ('text', 'item'),
        }
        if self.task not in task_to_modalities:
            raise ValueError(f"Unsupported pair task: {self.task}")
        return task_to_modalities[self.task]

    def _process_data(self):

        data_pair = []
        source_modality, target_modality = self._resolve_pair_modalities()
        for item_id in self.indices:
            source_index = self._resolve_modal_tokens(source_modality, item_id)
            target_index = self._resolve_modal_tokens(target_modality, item_id)
            data_pair.append([source_index, target_index])

        return data_pair


    def __len__(self):
        return len(self.data_pair)

    def __getitem__(self, index):

        d = self.data_pair[index]

        input = self.soft_prompt + d[0]
        output = d[1]

        return dict(input_ids=input, labels=output)

class FusionSeqRecDataset(BaseDataset):

    def __init__(self, args, task='seqitem2image', mode="train",
                 prompt_sample_num=1, prompt_id=0, sample_num=-1):
        super().__init__(args)

        self.mode = mode
        self.prompt_sample_num = prompt_sample_num
        self.prompt_id = prompt_id
        self.sample_num = sample_num
        self.task = task

        self.text_index_file = args.text_index_file
        self.image_index_file = args.image_index_file

        self.soft_prompt = args.soft_prompts[self.task]

        # load data
        self._load_data()
        # self._remap_items()

        # load data
        if self.mode == 'train':
            self.inter_data = self._process_train_data()
        elif self.mode == 'valid':
            self.inter_data = self._process_valid_data()
        elif self.mode == 'test':
            self.inter_data = self._process_test_data()
        else:
            raise NotImplementedError


    def _load_data(self):

        with open(os.path.join(self.data_path, self.dataset + ".inter.json"), 'r') as f:
            self.inters = json.load(f)
        with open(resolve_index_path(self.data_path, self.dataset, self.index_file), 'r') as f:
            self.indices = json.load(f)
        with open(resolve_index_path(self.data_path, self.dataset, self.text_index_file), 'r') as f:
            self.text_indices = json.load(f)
        with open(resolve_index_path(self.data_path, self.dataset, self.image_index_file), 'r') as f:
            self.image_indices = json.load(f)

    def _resolve_modal_tokens(self, modality, item_id):
        if modality == 'item':
            return "".join(self.indices[str(item_id)])
        if modality == 'text':
            return "".join(self.text_indices[str(item_id)])
        if modality == 'image':
            return "".join(self.image_indices[str(item_id)])
        raise ValueError(f"Unsupported modality: {modality}")

    def _resolve_seq_modalities(self):
        task_to_modalities = {
            'seqitem2image': ('item', 'image'),
            'seqimage2item': ('image', 'item'),
            'seqitem2text': ('item', 'text'),
            'seqtext2item': ('text', 'item'),
        }
        return task_to_modalities.get(self.task)

    def _resolve_history_tokens(self, source_modality, history, mix_modalities=False):
        resolved = []
        for hist_item in history:
            modality = source_modality
            if (
                mix_modalities
                and source_modality == "item"
                and self.fusion_source_modal_mix_prob > 0
                and random.random() < self.fusion_source_modal_mix_prob
            ):
                modality = random.choice(["image", "text"])
            resolved.append(self._resolve_modal_tokens(modality, hist_item))
        return resolved

    def _process_train_data(self):


        inter_data = []
        for uid in self.inters:
            items = self.inters[uid][:-2]
            for i in range(1, len(items)):
                one_data = dict()

                item = items[i]
                history = items[:i]
                if self.max_his_len > 0:
                    history = history[-self.max_his_len:]

                resolved_modalities = self._resolve_seq_modalities()
                if resolved_modalities is not None:
                    source_modality, target_modality = resolved_modalities
                    history = self._resolve_history_tokens(
                        source_modality,
                        history,
                        mix_modalities=True,
                    )
                    one_data["inters"] = ''.join(history)
                    one_data["item"] = self._resolve_modal_tokens(target_modality, item)

                else:
                    one_data["inters"] = history
                    one_data["item"] = item

                inter_data.append(one_data)

        return inter_data

    def _build_eval_row(self, row_idx, items, target_index):
        target_item = items[target_index]
        history = items[:target_index]
        if self.max_his_len > 0:
            history = history[-self.max_his_len:]

        resolved_modalities = self._resolve_seq_modalities()
        if resolved_modalities is not None:
            source_modality, target_modality = resolved_modalities
            history = self._resolve_history_tokens(source_modality, history)
            target = self._resolve_modal_tokens(target_modality, target_item)
            return {
                "inters": ''.join(history),
                "item": target,
                "user": row_idx,
            }

        return {
            "inters": history,
            "item": target_item,
            "user": row_idx,
        }

    def _sample_inter_data(self, inter_data):
        if self.sample_num > 0 and self.sample_num < len(inter_data):
            all_inter_idx = range(len(inter_data))
            sample_idx = np.random.choice(all_inter_idx, self.sample_num, replace=False)
            inter_data = np.array(inter_data)[sample_idx].tolist()
        eval_indices_file = getattr(self.args, "eval_indices_file", "")
        if eval_indices_file:
            indices = SeqRecDataset._load_eval_indices(eval_indices_file, len(inter_data))
            inter_data = [inter_data[index] for index in indices]
        eval_start = getattr(self.args, "eval_start", 0)
        eval_count = getattr(self.args, "eval_count", -1)
        eval_stride = getattr(self.args, "eval_stride", 1)
        if eval_start < 0:
            raise ValueError("--eval_start must be non-negative")
        if eval_count < -1:
            raise ValueError("--eval_count must be -1 or non-negative")
        if eval_stride <= 0:
            raise ValueError("--eval_stride must be positive")
        if eval_stride > 1:
            inter_data = inter_data[::eval_stride]
        if eval_start > 0 or eval_count >= 0:
            end = None if eval_count < 0 else eval_start + eval_count
            inter_data = inter_data[eval_start:end]
        return inter_data

    def _process_valid_data(self):
        inter_data = []
        for row_idx, uid in enumerate(self.inters):
            items = self.inters[uid]
            if len(items) < 2:
                continue
            inter_data.append(self._build_eval_row(row_idx, items, -2))
        return self._sample_inter_data(inter_data)

    def _process_test_data(self):
        inter_data = []
        for row_idx, uid in enumerate(self.inters):
            items = self.inters[uid]
            if len(items) < 1:
                continue
            inter_data.append(self._build_eval_row(row_idx, items, -1))
        return self._sample_inter_data(inter_data)

    def get_all_items(self):
        if self.all_items is not None:
            return self.all_items

        resolved_modalities = self._resolve_seq_modalities()
        if resolved_modalities is None:
            return super().get_all_items()

        _, target_modality = resolved_modalities
        if target_modality == "image":
            source = self.image_indices
        elif target_modality == "text":
            source = self.text_indices
        else:
            source = self.indices

        self.all_items = {"".join(index) for index in source.values()}
        return self.all_items

    def set_prompt(self, prompt_id):

        self.prompt_id = prompt_id

    def __len__(self):
        return len(self.inter_data)

    def __getitem__(self, index):

        d = self.inter_data[index]

        if self._resolve_seq_modalities() is not None:
            input = self.soft_prompt + d['inters']
            output = d['item']

        else:

            if random.random() > 0.5:
                history = ["".join(self.indices[str(i)]) for i in d['inters']]
                input = self.soft_prompt + ''.join(history)
                output = ''.join(self.image_indices[str(d['item'])])
            else:
                history = ["".join(self.image_indices[str(i)]) for i in d['inters']]
                input = self.soft_prompt + ''.join(history)
                output = ''.join(self.indices[str(d['item'])])

        return dict(input_ids=input, labels=output, label=d.get("user", index))

class SynergisticDataset(BaseDataset):
    """
    适配协同信息对比学习任务的 Dataset
    支持两个子任务：
    1. masktext: 预测 Text ID，单模态输入为 Text History
    2. maskimg:  预测 Image ID，单模态输入为 Image History
    """

    def __init__(self, args, task='masktext', mode="train",
                 prompt_sample_num=1, prompt_id=0, sample_num=-1):
        super().__init__(args)

        self.mode = mode
        self.prompt_sample_num = prompt_sample_num
        self.prompt_id = prompt_id
        self.sample_num = sample_num


        self.task = task.lower()

        self.soft_prompt = args.soft_prompts.get(self.task, "")
        if self.soft_prompt == "":

            self.soft_prompt = args.soft_prompts.get('synergistic', "")

        self.image_index_file = args.image_index_file


        self._load_data()


        if self.mode == 'train':
            self.inter_data = self._process_train_data()
        elif self.mode == 'valid':
            self.inter_data = self._process_valid_data()
        elif self.mode == 'test':
            self.inter_data = self._process_test_data()
        else:
            raise NotImplementedError

    def _load_data(self):
        with open(os.path.join(self.data_path, self.dataset + ".inter.json"), 'r') as f:
            self.inters = json.load(f)
        with open(resolve_index_path(self.data_path, self.dataset, self.index_file), 'r') as f:
            self.indices = json.load(f)
        with open(resolve_index_path(self.data_path, self.dataset, self.image_index_file), 'r') as f:
            self.image_indices = json.load(f)

    def _process_train_data(self):
        inter_data = []
        for uid in self.inters:
            items = self.inters[uid][:-2]

            for i in range(1, len(items)):
                one_data = dict()
                target_item = items[i]
                history_items = items[:i]

                if self.max_his_len > 0:
                    history_items = history_items[-self.max_his_len:]

                bi_modal_list = []
                unimodal_list = []

                for item_id in history_items:
                    s_id = str(item_id)
                    text_tokens = "".join(self.indices[s_id])
                    img_tokens = "".join(self.image_indices[s_id])


                    bi_modal_list.append(text_tokens + img_tokens)


                    if self.task == 'masktext':
                        unimodal_list.append(text_tokens)
                    elif self.task == 'maskimg':
                        unimodal_list.append(img_tokens)

                one_data["inters_bi"] = "".join(bi_modal_list)
                one_data["inters_uni"] = "".join(unimodal_list)


                s_target = str(target_item)
                if self.task == 'masktext':
                    one_data["label"] = "".join(self.indices[s_target])
                elif self.task == 'maskimg':
                    one_data["label"] = "".join(self.image_indices[s_target])

                inter_data.append(one_data)

        return inter_data

    def _process_valid_data(self):
        inter_data = []
        for uid in self.inters:
            items = self.inters[uid]
            target_item = items[-2]
            history_items = items[:-2]

            if self.max_his_len > 0:
                history_items = history_items[-self.max_his_len:]

            bi_modal_list = []
            for item_id in history_items:
                s_id = str(item_id)

                bi_modal_list.append("".join(self.indices[s_id]) + "".join(self.image_indices[s_id]))

            one_data = dict()
            one_data["inters_bi"] = "".join(bi_modal_list)
            one_data["inters_uni"] = ""


            s_target = str(target_item)
            if self.task == 'masktext':
                one_data["label"] = "".join(self.indices[s_target])
            elif self.task == 'maskimg':
                one_data["label"] = "".join(self.image_indices[s_target])

            inter_data.append(one_data)
        return inter_data

    def _process_test_data(self):
        inter_data = []
        for uid in self.inters:
            items = self.inters[uid]
            target_item = items[-1]
            history_items = items[:-1]

            if self.max_his_len > 0:
                history_items = history_items[-self.max_his_len:]

            bi_modal_list = []
            for item_id in history_items:
                s_id = str(item_id)
                bi_modal_list.append("".join(self.indices[s_id]) + "".join(self.image_indices[s_id]))

            one_data = dict()
            one_data["inters_bi"] = "".join(bi_modal_list)
            one_data["inters_uni"] = ""

            s_target = str(target_item)
            if self.task == 'masktext':
                one_data["label"] = "".join(self.indices[s_target])
            elif self.task == 'maskimg':
                one_data["label"] = "".join(self.image_indices[s_target])

            inter_data.append(one_data)

        if self.sample_num > 0:
            all_inter_idx = range(len(inter_data))
            sample_idx = np.random.choice(all_inter_idx, self.sample_num, replace=False)
            inter_data = np.array(inter_data)[sample_idx].tolist()

        return inter_data

    def __len__(self):
        return len(self.inter_data)

    def __getitem__(self, index):
        d = self.inter_data[index]


        input_ids = self.soft_prompt + d['inters_bi']


        if self.mode == 'train':

            unimodal_input_ids = self.soft_prompt + d['inters_uni']
        else:
            unimodal_input_ids = ""

        labels = d['label']

        return dict(
            input_ids=input_ids,
            labels=labels,
            unimodal_input_ids=unimodal_input_ids,
            label=index
        )
