import os
import json
import logging
import torch
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from TTS.api import TTS
from tenacity import retry, stop_after_attempt, wait_exponential
import hashlib
import warnings
warnings.filterwarnings("ignore")

# 配置参数
INPUT_FOLDER = "dev"
OUTPUT_FOLDER = "audio_dev._coqui"
# 配置参数
BATCH_SIZE = 32  # 根据你的内存大小调整
SAVE_INTERVAL = max(BATCH_SIZE // 10, 1)  # 保存频率随batch size调整

PROGRESS_FILE = 'tts_progress.json'
MAX_SAMPLES = 127  # 新增：限制处理的样本数量

# 创建输出文件夹
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# 配置日志
logging.basicConfig(
    filename='tts_conversion.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.ERROR
)
logging.getLogger("TTS").setLevel(logging.ERROR)
logging.getLogger("numba").setLevel(logging.WARNING)

# 全局TTS模型
tts = None

def init_tts():
    """初始化TTS模型"""
    global tts
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        @retry(stop=stop_after_attempt(3), 
               wait=wait_exponential(multiplier=1, min=4, max=10))
        def download_and_init_tts():
            return TTS(
                model_name="tts_models/en/ljspeech/tacotron2-DDC",
                progress_bar=False,
                gpu=torch.cuda.is_available()  # 使用 gpu 参数代替 .to()
            )
        
        print("正在初始化TTS模型，首次使用需要下载模型文件...")
        tts = download_and_init_tts()
        logging.info(f"TTS model initialized on {device}")
        
    except Exception as e:
        logging.error(f"Failed to initialize TTS model: {str(e)}")
        print(f"初始化TTS模型失败: {str(e)}")
        print("请检查网络连接，确保可以访问模型下载地址")
        raise


def get_filename(dialogue_id, turn_index, speaker, utterance):
    """生成唯一的文件名"""
    hash_object = hashlib.md5(utterance.encode('utf-8'))
    hash_hex = hash_object.hexdigest()
    return f"dialogue_{dialogue_id}_turn_{turn_index}_{speaker}_{hash_hex[:8]}.wav"

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def process_utterance(args):
    """处理单个utterance并保存为音频文件"""
    global tts
    dialogue_id, turn_index, speaker, utterance = args
    if not utterance:
        return

    filename = get_filename(dialogue_id, turn_index, speaker, utterance)
    output_path = os.path.join(OUTPUT_FOLDER, filename)

    if os.path.exists(output_path):
        return filename

    try:
        if tts is None:
            init_tts()
        tts.tts_to_file(
            text=utterance,
            file_path=output_path,
            gpu=torch.cuda.is_available()  # 确保这里也使用 gpu 参数
        )
        return filename
    except Exception as e:
        logging.error(f"Error converting text to speech for file {filename}: {e}")
        raise

def validate_utterance(utterance):
    """验证utterance是否有效"""
    if not isinstance(utterance, str):
        return False
    if len(utterance.strip()) == 0:
        return False
    if len(utterance) > 500:  # 避免过长的文本
        return False
    return True

def process_json_file(json_path, dialogues_processed=0):
    """处理单个JSON文件，限制处理的对话数量"""
    if not os.path.exists(json_path):
        logging.error(f"File not found: {json_path}")
        return [], dialogues_processed
        
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                logging.error(f"Error decoding JSON from file {json_path}: {e}")
                return [], dialogues_processed
    except IOError as e:
        logging.error(f"Error reading file {json_path}: {e}")
        return [], dialogues_processed

    utterances = []
    for dialogue in data:
        if dialogues_processed >= MAX_SAMPLES:  # 检查对话数量而不是utterance数量
            break
            
        dialogue_id = dialogue.get("dialogue_id", "")
        if not dialogue_id:
            continue
            
        turns = dialogue.get("turns", [])
        dialogue_utterances = []
        for idx, turn in enumerate(turns):
            speaker = turn.get("speaker", "UNKNOWN")
            utterance = turn.get("utterance", "").strip()
            if validate_utterance(utterance):
                dialogue_utterances.append((dialogue_id, idx, speaker, utterance))
        
        if dialogue_utterances:  # 如果这个对话有有效的utterance
            utterances.extend(dialogue_utterances)
            dialogues_processed += 1  # 每处理完一个完整对话才增加计数
                
    return utterances, dialogues_processed

def save_progress(processed_files):
    """保存处理进度"""
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(list(processed_files), f)

def load_progress():
    """加载处理进度"""
    try:
        with open(PROGRESS_FILE, 'r') as f:
            return set(json.load(f))
    except FileNotFoundError:
        return set()

# def process_batch(utterances):
#     """处理一批utterances"""
#     if not utterances:
#         return
        
#     num_workers = max(cpu_count() - 1, 1)
#     print(f"处理 {len(utterances)} 个utterances，使用 {num_workers} 个进程")
    
#     try:
#         with Pool(processes=num_workers, initializer=init_tts) as pool:
#             with tqdm(total=len(utterances), desc="Converting to audio", unit="utterance") as pbar:
#                 for _ in pool.imap(process_utterance, utterances):
#                     pbar.update(1)
#     except Exception as e:
#         logging.error(f"批处理过程中发生错误: {e}")
#         raise

# 3. 优化进程数
def process_batch(utterances):
    global processed_files
    num_workers = min(4, max(cpu_count() - 1, 1))
    
    try:
        # 使用单进程处理，因为TTS模型已经是全局的
        for i, utterance in enumerate(tqdm(utterances, desc="Converting to audio")):
            try:
                process_utterance(utterance)
                if (i + 1) % SAVE_INTERVAL == 0:
                    save_progress(processed_files)
            except Exception as e:
                logging.error(f"处理utterance时出错: {e}")
                continue
                
    except Exception as e:
        logging.error(f"批处理过程中发生错误: {e}")
        raise


def cleanup():
    """清理临时文件和资源"""
    try:
        # 可以添加清理临时文件的代码
        pass
    except Exception as e:
        logging.error(f"清理过程中发生错误: {e}")

def main():
    """主函数"""
    try:
        processed_files = load_progress()
        
        json_files = [f for f in os.listdir(INPUT_FOLDER) 
                     if f.endswith('.json') and f not in processed_files]
                     
        if not json_files:
            print(f"在 {INPUT_FOLDER} 中没有新的JSON文件需要处理。")
            return
            
        print(f"找到 {len(json_files)} 个新的JSON文件需要处理。")

        # 初始化TTS模型
        init_tts()

        all_utterances = []
        dialogues_processed = 0  # 改名以更清晰地表示是对话数量
        
        with tqdm(total=len(json_files), desc="收集utterances", unit="file") as pbar:
            for json_file in json_files:
                if dialogues_processed >= MAX_SAMPLES:
                    break
                    
                json_path = os.path.join(INPUT_FOLDER, json_file)
                utterances, dialogues_processed = process_json_file(json_path, dialogues_processed)
                all_utterances.extend(utterances)
                
                if len(all_utterances) >= BATCH_SIZE:
                    process_batch(all_utterances)
                    all_utterances = []
                
                processed_files.add(json_file)
                save_progress(processed_files)
                pbar.update(1)
            
            if all_utterances:
                process_batch(all_utterances)

        print(f"处理完成 {dialogues_processed} 个对话。")
        print(f"音频文件保存在 '{OUTPUT_FOLDER}' 文件夹中。")

    except Exception as e:
        logging.error(f"处理过程中发生错误: {e}")
        raise

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"程序执行出错: {e}")
        logging.error(f"程序执行出错: {e}")
    finally:
        cleanup()