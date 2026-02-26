import json
import multiprocessing
from functools import partial
from pathlib import Path

def process_item(item):
    """处理单个item: 添加image字段"""
    if "image_id" in item:
        item["image"] = f"{item['image_id']}.JPEG"  # 自动添加.JPEG后缀
    return item

def process_chunk(chunk, input_file, output_file):
    """处理一个数据块"""
    with open(input_file, 'r') as fin, open(output_file, 'a') as fout:
        for line in chunk:
            item = json.loads(line)
            processed_item = process_item(item)
            fout.write(json.dumps(processed_item) + '\n')

def parallel_process_jsonl(input_path, output_path, num_processes=None):
    """并行处理JSONL文件"""
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()  # 默认使用全部CPU核心
    
    # 确保输出目录存在
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # 清空输出文件（如果已存在）
    with open(output_path, 'w') as f:
        pass
    
    # 分割文件为多个块
    with open(input_path, 'r') as f:
        lines = f.readlines()
    chunk_size = len(lines) // num_processes
    
    # 创建进程池
    with multiprocessing.Pool(num_processes) as pool:
        chunks = [lines[i:i + chunk_size] for i in range(0, len(lines), chunk_size)]
        pool.starmap(
            partial(process_chunk, input_file=input_path, output_file=output_path),
            [(chunk,) for chunk in chunks]
        )

if __name__ == '__main__':
    input_file = "./infoseek/infoseek_val_withkb_article_retriever.jsonl"    # 输入文件路径
    output_file = "./infoseek/infoseek_val_withkb_image_article_retriever.jsonl"  # 输出文件路径
    
    # 执行并行处理（默认使用全部CPU核心）
    parallel_process_jsonl(input_file, output_file)
