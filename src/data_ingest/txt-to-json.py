import pandas as pd
import re
import json
from pathlib import Path

def parse_txt_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        raw_text = f.read()
    
    # 以 --- 分篇
    texts = [part.strip() for part in raw_text.split('---') if part.strip()]
    return texts

def parse_texts(texts):
    results = []
    pianhao_counter = 1  # 篇號從1開始
    
    for text in texts:
        lines = text.strip().splitlines()
        fupan = ""
        fuji = ""
        content_lines = []
        
        # 讀取賦篇和賦家
        for line in lines:
            if line.startswith("賦篇："):
                fupan = line.replace("賦篇：", "").strip()
            elif line.startswith("賦家："):
                fuji = line.replace("賦家：", "").strip()
            else:
                content_lines.append(line.strip())

        # 整理段落
        parsed_text = {
            "篇號": pianhao_counter,
            "賦篇": fupan,
            "賦家": fuji,
            "段落": []
        }
        
        # 新增累加計數器
        juanzu_global_idx = 1
        juzi_global_idx = 1
        
        for duan_idx, paragraph in enumerate(content_lines, 1):
            if not paragraph:
                continue
            # 以「。」切句組
            sentence_groups = re.split(r'(?<=。)', paragraph)
            sentence_groups = [sg for sg in sentence_groups if sg.strip()]

            duanluo = {
                "段落編號": duan_idx,
                "句組": []
            }

            for group in sentence_groups:
                group = group.strip()
                # 句組內用 ，；、：。 切成句子
                sentences = re.split(r'[，；、：。]', group)
                sentences = [s for s in sentences if s.strip()]
                
                juanzu = {
                    "句組編號": juanzu_global_idx,
                    "句子": []
                }
                
                for sentence in sentences:
                    juanzu["句子"].append({
                        "句編號": juzi_global_idx,
                        "內容": sentence.strip()
                    })
                    juzi_global_idx += 1  # 句子累加
                
                duanluo["句組"].append(juanzu)
                juanzu_global_idx += 1  # 句組累加
            
            parsed_text["段落"].append(duanluo)
        
        results.append(parsed_text)
        pianhao_counter += 1  # 下一篇篇號加1
    
    return results

def flatten_to_df(parsed_results):
    rows = []
    for article in parsed_results:
        pianhao = article['篇號']
        fupan = article['賦篇']
        fuji = article['賦家']
        for duan in article['段落']:
            duan_idx = duan['段落編號']
            for juanzu in duan['句組']:
                group_idx = juanzu['句組編號']
                for ju in juanzu['句子']:
                    ju_idx = ju['句編號']
                    content = ju['內容']
                    rows.append({
                        "篇號": pianhao,
                        "賦篇": fupan,
                        "賦家": fuji,
                        "段落編號": duan_idx,
                        "句組編號": group_idx,
                        "句編號": ju_idx,
                        "內容": content
                    })
    df = pd.DataFrame(rows)
    return df

# === 主程式 ===
# 請換成你的檔案路徑
file_path = r"D:\lufu_allusion\data\raw\origin-text.txt"

# 執行流程
texts = parse_txt_file(file_path)
parsed_results = parse_texts(texts)

# 儲存路徑
output_dir = Path(r"D:\lufu_allusion\data\processed")
output_dir.mkdir(parents=True, exist_ok=True)  # 如果資料夾不存在，自動建立

# 要儲存的檔案名稱
output_file = output_dir / "parsed_results.json"

# 把 parsed_results 儲存成 JSON
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(parsed_results, f, ensure_ascii=False, indent=2)

print(f"檔案已儲存到：{output_file}")

# 顯示結果
# df = flatten_to_df(parsed_results)
# print(df)
