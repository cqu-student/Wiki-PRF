import json
import os
from tqdm import tqdm
def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

# 将数据按 data_id 构建索引
def build_index(data):
    index = {}
    for item in data:
        index[item['data_id']] = item
    return index

def build_wiki_index(data):
    index = {}
    for item in data:
        index[item['wikidata_id']] = item
    return index


# 合并两个文件的数据
def merge_data(infoseek_train, infoseek_train_withkb, image_root, wiki_data=None):
    train_index = build_index(infoseek_train)
    withkb_index = build_index(infoseek_train_withkb)
    if wiki_data is not None:
        wiki_data = build_wiki_index(wiki_data)
    merged_data = []
    image_formats = ['JPEG', 'jpg', 'png']
    # 遍历 infoseek_train 数据并合并
    findnot_image = 0
    for data_id, item in tqdm(train_index.items()):
        if data_id in withkb_index:
            # 合并两个字典
            image_id = item['image_id']
            
            for image_format in image_formats:
                image_file_name = f"{image_id}.{image_format}"
                if os.path.exists(os.path.join(image_root, image_file_name)):
                    item['image'] = image_file_name
                    break
            if 'image' not in item:
                # raise FileNotFoundError(f"Image file not found for data_id {data_id}")
                print(f"Image file not found for data_id {data_id}")
                findnot_image += 1
                continue
            merged_item = {**item, **withkb_index[data_id]}
            
            if wiki_data is not None:
                # 添加 wiki 数据
                entity_id = withkb_index[data_id]['entity_id']
                wiki_item = wiki_data[entity_id]
                merged_item['entity_context'] = wiki_item['wikipedia_content']
            
            merged_data.append(merged_item)
        else:
            # 如果没有匹配的 data_id，则略过
            # merged_data.append(item)
            pass
    
    # 处理 infoseek_train_withkb 中未匹配的部分
    # for data_id, item in withkb_index.items():
    #     if data_id not in train_index:
    #         merged_data.append(item)
    print(f"findnot_image: {findnot_image}")
    return merged_data

# 保存合并后的数据到 Python 文件
def save_to_python_file(data, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(f"{json.dumps(item, ensure_ascii=False)}\n")

# 主函数
def main(infoseek_train_path, infoseek_train_withkb_path, image_root, output_file, wiki_path=None):

    # 读取数据
    infoseek_train = read_jsonl(infoseek_train_path)
    infoseek_train_withkb = read_jsonl(infoseek_train_withkb_path)
    
    if wiki_path is not None:
        wiki_data = read_jsonl(wiki_path)
    else:
        wiki_data = None
    
    # 合并数据
    merged_data = merge_data(infoseek_train, infoseek_train_withkb, image_root, wiki_data=wiki_data)
    
    # 保存到 Python 文件
    save_to_python_file(merged_data, output_file)
    print(f"Merged data has been saved to {output_file}")

if __name__ == "__main__":
    # 文件路径
    infoseek_train_path = "./infoseek/infoseek_train.jsonl"
    infoseek_train_withkb_path = "./infoseek/infoseek_train_withkb.jsonl"
    wiki_path = "./infoseek/Wiki6M_ver_1_0.jsonl"
    output_file = "./infoseek/infoseek_train_withkb_wkbcontent_merged.jsonl"
    image_root = "./oven/oven_images"
    
    main(infoseek_train_path=infoseek_train_path, infoseek_train_withkb_path=infoseek_train_withkb_path, image_root=image_root, output_file=output_file, wiki_path=wiki_path)
