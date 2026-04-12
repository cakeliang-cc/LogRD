import re
import csv
from tqdm import tqdm
from preprocess import clean


def extract_block_ids(line):
    """提取日志行中的所有 blk_ 号"""
    # 匹配 blk_ 后面跟着数字的模式
    pattern = r'blk_[-]?\d+'
    block_ids = re.findall(pattern, line)
    return block_ids


def count_lines(file_path):
    """统计文件总行数"""
    print("正在统计文件行数...")
    count = 0
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for _ in tqdm(f, desc="统计行数"):
            count += 1
    return count


def process_hdfs_log(input_file, output_file):
    """
    处理 HDFS.log 文件，对每行执行 clean 操作并提取 blk_ 号
    
    Parameters
    ----------
    input_file: str, 输入日志文件路径
    output_file: str, 输出 CSV 文件路径
    """
    # 统计总行数
    total_lines = count_lines(input_file)
    print(f"文件总行数: {total_lines}")
    
    # 处理文件
    print("正在处理文件...")
    with open(input_file, 'r', encoding='utf-8', errors='ignore') as f_in, \
         open(output_file, 'w', encoding='utf-8', newline='') as f_out:
        
        writer = csv.writer(f_out)
        # 写入 CSV 表头
        writer.writerow(['original_line', 'block_ids', 'content'])
        
        # 使用 tqdm 显示进度
        for line in tqdm(f_in, total=total_lines, desc="处理进度"):
            line = line.strip()
            if not line:  # 跳过空行
                continue
            
            # 提取 blk_ 号
            block_ids = extract_block_ids(line)
            block_ids_str = ', '.join(block_ids) if block_ids else ''
            
            # 执行 clean 操作
            cleaned_content = clean(line)
            
            # 写入 CSV
            writer.writerow([line, block_ids_str, cleaned_content])
    
    print(f"\n处理完成！结果已保存到: {output_file}")


if __name__ == '__main__':
    input_file = '/root/autodl-tmp/versionHDFS/LogBGL3/output/HDFS.log'
    output_file = '/root/autodl-tmp/versionHDFS/LogBGL3/output/hdfs/hdfs_processed.csv'
    process_hdfs_log(input_file, output_file)

