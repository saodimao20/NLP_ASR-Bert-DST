import os
import json
import torch
import random
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.char as nac
import re
from typing import List, Dict

def get_file_number(filename):
    """
    从文件名中提取数字部分。例如，dialogues_001.json -> 1
    """
    basename = os.path.splitext(filename)[0]
    parts = basename.split('_')
    if len(parts) < 2:
        return None
    try:
        number = int(parts[1].lstrip('0'))
        return number
    except ValueError:
        return None

def add_common_asr_errors(text: str) -> str:
    """
    模拟常见的ASR错误，包括同音字替换和语音混淆
    """
    # 同音字替换字典
    homophones = {
        "there": ["their", "they're"],
        "to": ["too", "two"],
        "four": ["for", "fore"],
        "write": ["right", "rite"],
        "hear": ["here", "hair"],
        "your": ["you're", "yore"],
        "its": ["it's"],
        "weather": ["whether"],
        "which": ["witch"],
        "who's": ["whose"],
        "accept": ["except"],
        "affect": ["effect"]
    }
    
    # 语音混淆字典
    phonetic_errors = {
        "s": ["z", "c"],
        "f": ["th", "ph"],
        "k": ["c", "q"],
        "m": ["n"],
        "d": ["t"],
        "b": ["p"],
        "v": ["f"],
        "g": ["j"],
        "ch": ["sh", "tch"],
        "ai": ["ay", "ei"]
    }

    # 分词处理
    words = text.split()
    modified_words = []

    for word in words:
        # 随机决定是否对当前词进行修改
        if random.random() < 0.4:  # 30%的概率进行修改
            # 检查同音字替换
            lower_word = word.lower()
            if lower_word in homophones:
                word = random.choice(homophones[lower_word])
            else:
                # 应用语音混淆
                for sound, alternatives in phonetic_errors.items():
                    if sound in lower_word and random.random() < 0.4:  # 20%的概率进行音素替换
                        word = word.replace(sound, random.choice(alternatives))
        
        modified_words.append(word)

    return " ".join(modified_words)

def simulate_asr_errors(text: str, augmenter, aug_p=0.3) -> str:
    """
    组合多种ASR错误模拟方法
    """
    # 首先应用nlpaug增强
    augmented_text = augmenter.augment(text)[0]
    
    # 然后应用常见ASR错误模拟
    final_text = add_common_asr_errors(augmented_text)
    
    return final_text

def process_file(input_path: str, output_path: str, augmenter):
    """
    处理单个JSON文件，修改其中的utterance字段，并保存到输出路径。
    """
    with open(input_path, 'r', encoding='utf-8') as infile:
        try:
            data = json.load(infile)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from {input_path}: {e}")
            return
    
    # 遍历每个对话
    for dialogue in data:
        dialogue_id = dialogue.get("dialogue_id", "")
        if not dialogue_id:
            continue
        
        # 提取对话ID中的数字
        dialogue_number_str = dialogue_id.split('_')[0]
        try:
            dialogue_number = int(dialogue_number_str)
        except ValueError:
            print(f"Invalid dialogue_id format: {dialogue_id}")
            continue
        
        # 检查文件名中的数字
        filename = os.path.basename(input_path)
        file_number = get_file_number(filename)
        if file_number is None:
            print(f"Could not extract file number from filename: {filename}")
            continue
        
        if dialogue_number != file_number:
            print(f"Dialogue ID {dialogue_id} does not match file number {file_number} in {filename}")
        
        # 处理每个轮次的utterance
        for turn in dialogue.get("turns", []):
            utterance = turn.get("utterance", "")
            if utterance:
                augmented_utterance = simulate_asr_errors(utterance, augmenter)
                turn["utterance"] = augmented_utterance
    
    # 保存修改后的数据
    with open(output_path, 'w', encoding='utf-8') as outfile:
        json.dump(data, outfile, ensure_ascii=False, indent=2)

def main():
    input_folder = 'test'
    output_folder = 'test_augmented'
    os.makedirs(output_folder, exist_ok=True)
    
    # 配置增强器
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    
    # 创建多个增强器
    augmenters = []
    
    # 添加字符级别的增强器
    # RandomCharAug的正确参数设置
    char_aug = nac.RandomCharAug(
        action="substitute",
        aug_char_p=0.2,  # 替换为aug_char_p
        aug_word_p=0.3   # 添加aug_word_p参数
    )
    augmenters.append(char_aug)
    
    # 尝试添加上下文词嵌入增强器
    try:
        word_aug = naw.ContextualWordEmbsAug(
            model_path='bert-base-uncased',
            action="substitute",
            device=device,
            aug_p=0.2
        )
        augmenters.append(word_aug)
    except Exception as e:
        print(f"Could not initialize ContextualWordEmbsAug: {e}")
    
    # 选择主要增强器
    primary_augmenter = augmenters[0] if augmenters else char_aug
    
    # 处理所有文件
    for filename in os.listdir(input_folder):
        if filename.endswith('.json'):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            print(f"Processing {filename}...")
            process_file(input_path, output_path, primary_augmenter)
    
    print("ASR增强完成。增强后的文件保存在 'dev_augmented' 文件夹中。")

if __name__ == "__main__":
    main()