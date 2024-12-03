import os
import json
from googletrans import Translator
from tqdm import tqdm
import copy
import time

def back_translate(text, translator, src_lang='en', intermediate_lang='zh-cn'):
    """
    将文本翻译到中间语言，再翻译回原始语言，实现回译。
    """
    try:
        if not isinstance(text, str) or not text.strip():
            return text
            
        time.sleep(1)  # 添加延时避免请求过快
        # 翻译到中间语言
        translated = translator.translate(text, src=src_lang, dest=intermediate_lang)
        if not translated or not translated.text:
            return text
            
        time.sleep(1)  # 添加延时避免请求过快
        # 再翻译回原始语言
        back_translated = translator.translate(translated.text, src=intermediate_lang, dest=src_lang)
        return back_translated.text if back_translated else text
    except Exception as e:
        print(f"翻译出错: {e}")
        return text

def process_file(input_filepath, output_filepath, translator):
    """
    处理单个JSON文件，进行回译并添加噪声。
    """
    try:
        with open(input_filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        new_data = copy.deepcopy(data)
        
        for dialogue in tqdm(new_data, desc=f"Processing {os.path.basename(input_filepath)}"):
            if 'turns' not in dialogue:
                continue
                
            for turn in dialogue['turns']:
                if 'utterance' not in turn:
                    continue
                    
                utterance = turn.get('utterance', '')
                if utterance and isinstance(utterance, str):
                    noisy_utterance = back_translate(utterance, translator)
                    turn['utterance_noisy'] = noisy_utterance

        with open(output_filepath, 'w', encoding='utf-8') as f:
            json.dump(new_data, f, ensure_ascii=False, indent=2)
            
    except Exception as e:
        print(f"处理文件 {input_filepath} 时出错: {e}")

def main():
    base_dir = os.getcwd()
    input_folder = os.path.join(base_dir, 'dev')
    output_folder = os.path.join(base_dir, 'dev_noisy')
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 使用最新版本的 googletrans
    translator = Translator()
    
    # 获取所有 JSON 文件并排序
    files = [f for f in os.listdir(input_folder) if f.endswith('.json')]
    files.sort()
    
    print("将按顺序处理以下文件：", files)
    
    for filename in files:
        try:
            input_filepath = os.path.join(input_folder, filename)
            output_filepath = os.path.join(output_folder, filename)
            print(f"正在处理文件: {filename}")
            process_file(input_filepath, output_filepath, translator)
        except Exception as e:
            print(f"处理 {filename} 时出错: {e}")
            continue

if __name__ == "__main__":
    main()