from argparse import ArgumentParser
import json, tqdm
import torch
import os
from .model import (
    ClipRetriever,
    MistralAnswerGenerator,
    GPT4AnswerGenerator,
    reconstruct_wiki_article,
    PaLMAnswerGenerator,
    reconstruct_wiki_sections,
    WikipediaKnowledgeBaseEntry,
    BgeTextReranker,
)
from utils import load_csv_data, get_test_question, get_image, remove_list_duplicates
import PIL
def load_jsonl_data(test_file_path):
    import json

    result_list = []
    with open(test_file_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())  # [3,6](@ref)
            result_list.append(data)
    return result_list
def run_retrieval(test_file_path, 
                  knowledge_base_path, 
                  faiss_index_path,
                  **kwargs):
    test_list = load_jsonl_data(test_file_path)
    retriever = ClipRetriever(device="cuda:0", model=kwargs["retriever_vit"])
    # pdb.set_trace()
    retriever.load_knowledge_base(knowledge_base_path)
    retriever.load_faiss_index(faiss_index_path)
    test_retriever = []
    for i, test_example in tqdm.tqdm(enumerate(test_list)):
        #TODO: question
        # data_id = test_example["data_id"] 
        if os.path.exists(os.path.join(data_root, test_example['image_id'] + ".jpg")):
            image_path = os.path.join(data_root, test_example['image_id'] + ".jpg")
        elif os.path.exists(os.path.join(data_root, test_example['image_id'] + ".JPEG")):
            image_path = os.path.join(data_root, test_example['image_id'] + ".JPEG")
        image = PIL.Image.open(image_path)
        top_1 = retriever.retrieve_image_faiss(image, top_k=1)
        entry = top_1[0]["kb_entry"]
        entry_sections = reconstruct_wiki_sections(entry)
        test_example['entity_context'] = entry_sections
        test_retriever.append(test_example)
        # print(entry_sections)
    print("retriever done")
    with open(output_path, "w", encoding="utf-8") as f:
        for item in test_retriever:
            # 每行写入一个JSON对象，并处理中文编码 [6,9](@ref)
            line = json.dumps(item, ensure_ascii=False)
            f.write(line + "\n")  # 换行符分隔 [4,5](@ref)
    print("write done")
    
if __name__ == "__main__":
    test_file_path ="./infoseek/infoseek_val.jsonl"
    knowledge_base_path = "./infoseek/wiki_100_dict_v4.json"
    
    faiss_index_path = "./infoseek/infoseek_faiss/"
    data_root = "./oven/oven_images/"
    retriever_vit = "eva-clip"
    # output_path = "./infoseek/infoseek_train_withkb_retriever.jsonl"
    output_path = "./infoseek/infoseek_val_withkb_retriever.jsonl"
    run_retrieval(test_file_path, knowledge_base_path, faiss_index_path, retriever_vit=retriever_vit)

