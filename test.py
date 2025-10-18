from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
import torch
import json
from typing import Any, Callable, Optional, Union, Sized
import json
from tqdm import tqdm
import re
import os
from pprint import pprint
import random
from peft import PeftModel
from eval.answer_reward_utils import evaluate_example
from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config
from dataclasses import dataclass, field
from torch.utils.data import Dataset
import gc
import yaml
import time
import math
from PIL import Image
from torch.utils.data.distributed import DistributedSampler
from retriever import ClipRetriever
from answer_generator import reconstruct_wiki_article, reconstruct_wiki_sections

import torch.distributed as dist
import torch.multiprocessing as mp

def extract_bbox(data):
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError:
            return None

    if isinstance(data, list) and len(data) > 0:
        first_item = data[0]
        if isinstance(first_item, dict):
            return first_item.get("bbox_2d")
    elif isinstance(data, dict):
        return data.get("bbox_2d")

    return None

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'  # 泛化端口
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

class LazySupervisedDataset_wRAG(Dataset):
    def __init__(self, data_path: str, image_root: str):
        super(LazySupervisedDataset_wRAG, self).__init__()
        self.image_root = image_root
        self.list_data_dict = []

        if data_path.endswith(".yaml"):
            with open(data_path, "r") as file:
                yaml_data = yaml.safe_load(file)
                datasets = yaml_data.get("datasets")
                for data in datasets:
                    data_dict = data.get("data")
                    json_path = data_dict.get("json_path")
                    sampling_strategy = data_dict.get("sampling_strategy", "all")
                    sampling_number = data_dict.get("sampling_number", None)
                    if json_path.endswith(".jsonl"):
                        cur_data_dict = []
                        with open(json_path, "r") as json_file:
                            for line in json_file:
                                cur_data_dict.append(json.loads(line.strip()))
                    else:
                        raise ValueError(f"Unsupported file type: {json_path}")

                    if ":" in sampling_strategy:
                        sampling_strategy, sampling_number = sampling_strategy.split(":")
                        if "%" in sampling_number:
                            sampling_number = math.ceil(int(sampling_number.split("%")[0]) * len(cur_data_dict) / 100)
                        else:
                            sampling_number = int(sampling_number)

                    if sampling_strategy == "first" and sampling_number is not None:
                        cur_data_dict = cur_data_dict[:sampling_number]
                    elif sampling_strategy == "end" and sampling_number is not None:
                        cur_data_dict = cur_data_dict[-sampling_number:]
                    elif sampling_strategy == "random" and sampling_number is not None:
                        random.shuffle(cur_data_dict)
                        cur_data_dict = cur_data_dict[:sampling_number]
                    print(f"Loaded {len(cur_data_dict)} samples from {json_path}")
                    self.list_data_dict.extend(cur_data_dict)
        else:
            raise ValueError(f"Unsupported file type: {data_path}")

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i):
        example = self.list_data_dict[i]
        document = example['entity_context']
        
        image_root = self.image_root
        if 'image' in example:
            image_path = os.path.join(image_root, example['image'])
            if not os.path.exists(image_path):
                image_path = os.path.join(image_root, example['image'].replace('.JPEG', '.jpg'))
            while not os.path.exists(image_path):
                print(f"Warning: Image not found, randomly selecting another.")
                new_index = random.randint(0, len(self.list_data_dict)-1)
                example = self.list_data_dict[new_index]
                image_path = os.path.join(image_root, example['image'])
            image = Image.open(image_path).convert("RGB")
        else:
            image = None

        return {
            'image': image,
            'problem': example['question'],
            'solution': example['answer_eval'],
            'document': document,
        }

def make_pre_conversation_grounding_retrieval(search):
    return { 
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Locate {object}, output its bbox coordinates using JSON format.".format(object=search)},
        ],
    }

def make_pre_conversation_caption_retrieval(question, caption):
    return { 
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Here is the question: {Question}. Here is the caption: {Caption}. Please combine them to generate a new concise caption.".format(Question=question, Caption=caption)},
        ],
    }

def evaluate_batch(rank, model, model1, retriever_text_actor, processor, batch_messages, batch_images, batch_question, batch_document, max_new_tokens=1024):
    caption_count = 0
    grounding_count = 0
    caption_time = 0.0
    grounding_time = 0.0
    filter_time = 0.0
    answer_time = 0.0

    text = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batch_messages]
    inputs = processor(text=text, images=batch_images, padding=True, padding_side="left", return_tensors="pt")
    inputs = inputs.to(model.device)

    with torch.no_grad():
        generated_ids = model1.generate(**inputs, use_cache=True, max_new_tokens=max_new_tokens, do_sample=False)
    
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    completions_first = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    del generated_ids_trimmed

    pattern = r'<tool>(.*?:.*?)</tool>'
    matches_all = [re.search(pattern, completion, re.DOTALL) for completion in completions_first]
    pattern1 = r'<tool>(.*?)</tool>'
    matches = [re.search(pattern1, completion, re.DOTALL) for completion in completions_first]

    search_result = []
    for i in range(len(matches_all)):
        if matches_all[i] is None:
            search_result.append("None")
            continue

        sections_search = []
        for line in matches[i].group(1).strip().split('\n'):
            match = re.match(r'\d+\.\s*(\w+):\s*(.+)', line.strip())
            if match:
                key = match.group(1).lower()
                value = match.group(2).strip()
                if key == "caption":
                    try:
                        start_time = time.time()
                        prompt_caption = [make_pre_conversation_caption_retrieval(question=batch_question[i], caption=value)]
                        prompt_captioning = processor.apply_chat_template(prompt_caption, tokenize=False, add_generation_prompt=True)
                        caption_prompt_inputs = processor(text=prompt_captioning, images=batch_images[i], return_tensors="pt", padding=True, padding_side="left")
                        caption_prompt_inputs = caption_prompt_inputs.to(model.device)
                        caption_prompt_completion_ids = model.generate(**caption_prompt_inputs, use_cache=True, max_new_tokens=max_new_tokens, do_sample=False)
                        caption_length = caption_prompt_inputs["input_ids"].size(1)
                        completion_ids = caption_prompt_completion_ids[:, caption_length:]
                        caption_completions = processor.batch_decode(completion_ids, skip_special_tokens=True)
                        del caption_prompt_inputs

                        top_1 = retriever_text_actor.retrieve_text_faiss(caption_completions[0], top_k=3)
                        text_section = []
                        for item in top_1:
                            entry = item["kb_entry"]
                            entry_sections = reconstruct_wiki_sections(entry)
                            filtered_sections = [s.strip() for s in entry_sections if s and s.strip()]
                            text_section.extend(filtered_sections)
                        text_section = list(set(text_section)) 
                        sim = retriever_text_actor.similarity_section_text(caption_completions[0], text_section)
                        sorted_indices = torch.argsort(sim, descending=True)
                        top5_indices = sorted_indices[:3].cpu().numpy()
                        top5_texts = [text_section[i] for i in top5_indices]
                        sections_search.extend(top5_texts)
                        end_time = time.time()
                        caption_count += 1
                        caption_time += (end_time - start_time)
                    except Exception as e:
                        end_time = time.time()
                        caption_time += (end_time - start_time)
                        print(f"Error in caption: {str(e)}")

                elif key == "grounding":
                    start_time = time.time()
                    grounding_count += 1
                    prompt_ground = [make_pre_conversation_grounding_retrieval(value)]
                    prompt_grounding = processor.apply_chat_template(prompt_ground, tokenize=False, add_generation_prompt=True)
                    grounding_prompt_inputs = processor(text=prompt_grounding, images=batch_images[i], return_tensors="pt", padding=True, padding_side="left")
                    grounding_prompt_inputs = grounding_prompt_inputs.to(model.device)
                    grounding_prompt_completion_ids = model.generate(**grounding_prompt_inputs, use_cache=True, max_new_tokens=max_new_tokens, do_sample=False)
                    grounding_length = grounding_prompt_inputs["input_ids"].size(1)
                    grounding_completion_ids = grounding_prompt_completion_ids[:, grounding_length:]
                    grounding_completions = processor.batch_decode(grounding_completion_ids, skip_special_tokens=True)
                    del grounding_prompt_inputs

                    try:
                        json_str = grounding_completions[0].strip('```json\n').strip('```').strip()
                        data_bbox = json.loads(json_str)
                        bbox = data_bbox[0]["bbox_2d"]
                        if bbox:    
                            image = batch_images[i]
                            cropped_img = image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
                            cropped_img = cropped_img.resize((224,224), Image.Resampling.LANCZOS)
                            top_1 = retriever_text_actor.retrieve_image_faiss(cropped_img, top_k=7)
                            text_section = []
                            for item in top_1:
                                entry = item["kb_entry"]
                                entry_sections = reconstruct_wiki_sections(entry)
                                filtered_sections = [s.strip() for s in entry_sections if s and s.strip()]
                                text_section.extend(filtered_sections)
                            text_section = list(set(text_section))
                            sim = retriever_text_actor.similarity_section_image(cropped_img, text_section)
                            sorted_indices = torch.argsort(sim, descending=True)  
                            top5_indices = sorted_indices[:3].cpu().numpy()  
                            top5_texts = [text_section[i] for i in top5_indices]
                            sections_search.extend(top5_texts)
                            torch.cuda.empty_cache()
                            end_time = time.time()
                            grounding_time += (end_time - start_time)
                    except Exception as e:
                        end_time = time.time()
                        grounding_time += (end_time - start_time)
                        print(f"Error in grounding: {str(e)}")

                elif key == "flip":
                    try:
                        batch_images[i] = batch_images[i].transpose(Image.FLIP_LEFT_RIGHT)
                    except Exception as e:
                        print(f"Error in flip: {str(e)}")

        sections_search = list(set(sections_search))
        search = ".".join(sections_search)
        search_result.append(search)

    curr_search_template = (
        "Here is the user question: <question>{Question}</question>. "
        "Here is the relevant information retrieved through image retrieval: <retrieved_information>{Document}</retrieved_information>. "
        "Here is the relevant information through <tool>{Search}</tool>: <search_result>{Search_result}</search_result>. "
        "To obtain useful information, you must conduct reasoning inside <think></think> first. "
        "After reasoning, provide the filtered information inside <answer></answer>."
    )

    def make_pre_conversation_image_with_gt_retrieval(example, document, search, search_result):
        return {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": curr_search_template.format(Question=example, Document=document, Search=search, Search_result=search_result)},
            ],
        }

    prompt_new_text = [
        [make_pre_conversation_image_with_gt_retrieval(batch_question[i], batch_document[i], matches[i].group(1) if matches[i] else "None", search_result[i])]
        for i in range(len(batch_messages))
    ]
    text = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in prompt_new_text]
    inputs = processor(text=text, images=batch_images, padding=True, padding_side="left", return_tensors="pt")
    inputs = inputs.to(model.device)

    try:
        start_filter_time = time.time()
        with torch.no_grad():
            generated_ids = model1.generate(**inputs, use_cache=True, max_new_tokens=max_new_tokens, do_sample=False)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        batch_output_first = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        del generated_ids_trimmed
        end_filter_time = time.time()
        filter_time += (end_filter_time - start_filter_time)
    except Exception as e:
        end_filter_time = time.time()
        print(f"Error in first stage generation: {str(e)}")
        batch_output_first = [""] * len(batch_messages)
        filter_time += (end_filter_time - start_filter_time)

    AFTER_QUESTION_TEMPLATE = "Context: {Document}\nQuestion: {Question}\nShort answer:"

    def _make_after_conversation_image(question, document):
        pattern = r'<answer>(.*?)</answer>'
        match = re.search(pattern, document, re.DOTALL)
        document = match.group(1).strip() if match else "None"
        return {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": AFTER_QUESTION_TEMPLATE.format(Question=question, Document=document)},
            ],
        }

    second_stage_messages = [[_make_after_conversation_image(q, d)] for q, d in zip(batch_question, batch_output_first)]
    second_text = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in second_stage_messages]
    second_inputs = processor(text=second_text, images=batch_images, padding=True, padding_side="left", return_tensors="pt")
    second_inputs = second_inputs.to(model.device)

    try:
        start_answer_time = time.time()
        with torch.no_grad():
            second_generated_ids = model.generate(**second_inputs, use_cache=True, max_new_tokens=max_new_tokens, do_sample=False)
        second_generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(second_inputs.input_ids, second_generated_ids)]
        batch_output_final = processor.batch_decode(second_generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        end_answer_time = time.time()
        answer_time += (end_answer_time - start_answer_time)
    except Exception as e:
        end_answer_time = time.time()
        answer_time += (end_answer_time - start_answer_time)
        print(f"Error in second stage generation: {str(e)}")
        batch_output_final = [""] * len(batch_messages)

    return batch_output_final, caption_count, grounding_count, caption_time, grounding_time, filter_time, answer_time

def eval_RAG(rank, world_size, steps, dataset, MODEL_PATH, PEFT_MODEL_PATH, OUTPUT_PATH, BSZ, TEST_DATASETS):
    setup(rank, world_size)
    all_start_time = time.time()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    print(f"Rank {rank} started")
    device_id = rank % torch.cuda.device_count()
    torch.cuda.set_device(device_id)
    device = torch.device(f"cuda:{device_id}")

    caption_total = 0
    grounding_total = 0
    caption_total_time = 0.0
    grounding_total_time = 0.0
    filter_total_time = 0.0
    answer_total_time = 0.0

    # Initialize retriever (paths should be provided via config in real use)
    retriever_text_actor = ClipRetriever(device=device, model="eva-clip")
    retriever_text_actor.load_knowledge_base(knowledge_base_path="<PATH_TO_KNOWLEDGE_BASE_JSON>")
    retriever_text_actor.load_faiss_index(load_index_path="<PATH_TO_FAISS_INDEX>", gpu_choice=device_id)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).eval().to(device)
    peft_model = PeftModel.from_pretrained(model, PEFT_MODEL_PATH)
    model1 = peft_model.merge_and_unload().eval().to(device)
    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    QUESTION_TEMPLATE = (
        "Given a question whose answer is within a knowledge base, you need to utilize one or more tools to query the knowledge base. "
        "Choose from: 1. Caption: detailed description. 2. Grounding: identify core subject. 3. Flip: flip image. "
        "Enclose reasoning in <think></think> and tool calls in <tool></tool>. "
        "Here is the user question: {Question}."
    )

    batch_messages = []
    batch_images = []
    batch_questions = []
    batch_solution = []
    batch_document = []
    for x in dataset:
        batch_messages.append([{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": QUESTION_TEMPLATE.format(Question=x["problem"])}
            ]
        }])
        img = x["image"] if "image" in x else Image.open(x["image_path"])
        w, h = img.size
        if w < 28 or h < 28:
            if w < h:
                new_w, new_h = 28, int(h * (28 / w))
            else:
                new_h, new_w = 28, int(w * (28 / h))
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        batch_images.append(img)
        batch_questions.append(x["problem"])
        batch_solution.append(x["solution"])
        batch_document.append(x["document"])

    chunk_size = len(batch_messages) // world_size
    start_idx = rank * chunk_size
    end_idx = (rank + 1) * chunk_size if rank != world_size - 1 else len(batch_messages)
    local_messages = batch_messages[start_idx:end_idx]
    local_images = batch_images[start_idx:end_idx]
    local_questions = batch_questions[start_idx:end_idx]
    local_solution = batch_solution[start_idx:end_idx]
    local_document = batch_document[start_idx:end_idx]

    all_outputs = []
    if rank == 0:
        pbar = tqdm(total=len(local_messages), desc="Inference")

    for i in range(0, len(local_messages), BSZ):
        batch_output, caption_count, grounding_count, caption_time, grounding_time, filter_time, answer_time = evaluate_batch(
            rank, model, model1, retriever_text_actor, processor,
            local_messages[i:i+BSZ], local_images[i:i+BSZ],
            local_questions[i:i+BSZ], local_document[i:i+BSZ]
        )
        all_outputs.extend(batch_output)
        caption_total += caption_count
        grounding_total += grounding_count
        caption_total_time += caption_time
        grounding_total_time += grounding_time
        filter_total_time += filter_time
        answer_total_time += answer_time
        if rank == 0:
            pbar.update(len(batch_output))

    # Reduce metrics across ranks
    for tensor in [caption_total, grounding_total, caption_total_time, grounding_total_time, filter_total_time, answer_total_time]:
        t = torch.tensor([tensor], device=device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        # Overwrite with reduced value if needed (not used in output below)

    output_path = OUTPUT_PATH.format(DATASET=TEST_DATASETS[0], STEPS=steps)
    output_middle_path = output_path + f"_{rank}.json"
    with open(output_middle_path, 'w') as f:
        json.dump({
            "metadata": {"rank": rank, "num_samples": len(all_outputs)},
            "question": local_questions,
            "results": all_outputs,
            "solution": local_solution
        }, f)

    if rank == 0:
        all_end_time = time.time()
        all_time = all_end_time - all_start_time
        with open(output_middle_path, 'a') as f:
            f.write(f"\nTotal Time: {all_time:.2f} s\n")
            # Note: Full metrics aggregation would require gathering all rank outputs

    if rank == 0:
        pbar.close()

    cleanup()
    del model, model1, processor, retriever_text_actor
    torch.cuda.empty_cache()
    gc.collect()

def main():
    TEST_DATASETS = ['infoseek_test']  # Keep if public; else use '<DATASET_NAME>'
    IMAGE_ROOT = "<PATH_TO_IMAGE_ROOT>"
    DATA_ROOT = "<PATH_TO_DATA_CONFIG_YAML>"

    dataset = LazySupervisedDataset_wRAG(DATA_ROOT, IMAGE_ROOT)
    random.seed(42)

    world_size = torch.cuda.device_count()
    print(f"Using {world_size} GPUs for evaluation")

    steps = 600
    MODEL_PATH = "Qwen2.5-VL-3B-Instruct"
    PEFT_MODEL_PATH = "<PATH_TO_PEFT_CHECKPOINT>"
    OUTPUT_PATH = "results_{DATASET}_step{STEPS}.json"
    BSZ = 2

    mp.spawn(eval_RAG, args=(world_size, steps, dataset, MODEL_PATH, PEFT_MODEL_PATH, OUTPUT_PATH, BSZ, TEST_DATASETS), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
