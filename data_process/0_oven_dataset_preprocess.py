import os
import tarfile
import shutil
def extract_tar_files(file_path):
    # 定义要解压的文件名模式
    base_name = "shard"
    file_extension = ".tar"
    # 遍历 shard01.tar 到 shard09.tar
    for i in range(1, 9):  # 从1到8  
        # 格式化文件名，例如 shard01.tar, shard02.tar 等
        file_name = f"{base_name}{i:02d}{file_extension}"
        # 检查文件是否存在
        file_name = os.path.join(file_path, file_name)
        if not os.path.exists(file_name):
            print(f"文件 {file_name} 不存在，跳过解压。")
            continue
        # 尝试解压文件
        try:
            print(f"正在解压 {file_name}...")
            with tarfile.open(file_name, "r") as tar:
                tar.extractall()
            print(f"{file_name} 解压成功！")
        except Exception as e:
            print(f"解压 {file_name} 时出错：{e}")

def move_files_to_all(file_path, target_folder):
    # 定义源文件夹的命名模w式
    base_folder = "0"
    
    
    # 如果目标文件夹不存在，则创建
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
        print(f"已创建目标文件夹：{target_folder}")
    
    for i in range(1, 10):  # 从1到9
        folder_name = f"{base_folder}{i}"  # 格式化文件夹名，例如 01, 02 等
        folder_name = os.path.join(file_path, folder_name)
        # 检查文件夹是否存在
        if not os.path.exists(folder_name) or not os.path.isdir(folder_name):
            print(f"文件夹 {folder_name} 不存在或不是一个有效的目录，跳过。")
            continue
        
        # 获取文件夹中的所有文件
        try:
            files = [f for f in os.listdir(folder_name) if os.path.isfile(os.path.join(folder_name, f))]
            if not files:
                print(f"文件夹 {folder_name} 中没有文件，跳过。")
                continue
            
            # 移动文件到 all 文件夹
            for file in files:
                source_path = os.path.join(folder_name, file)
                target_path = os.path.join(target_folder, file)
                
                # 如果目标文件已存在，则跳过
                if os.path.exists(target_path):
                    print(f"文件 {file} 已存在于 {target_folder}，跳过。")
                    continue
                shutil.move(source_path, target_path)
                
        
        except Exception as e:
            print(f"处理文件夹 {folder_name} 时出错：{e}")


if __name__ == "__main__":
    file_path = './oven'
    target_folder = './oven/oven_images'
    extract_tar_files(file_path)
    move_files_to_all(file_path, target_folder)
