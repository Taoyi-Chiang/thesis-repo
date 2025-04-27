import os
import requests
import zipfile

# 本研究運用的對比文獻資料列表
categories = {
    "十三經": [
        "KR1d0026", # 儀禮 
        "KR1e0007", # 公羊傳
        "KR1a0001", # 周易
        "KR1d0001", # 周禮
        "KR1f0001", # 孝經
        "KR1h0001", # 孟子
        "KR1b0001", # 尚書
        "KR1e0001", # 左傳
        "KR1c0001", # 毛詩
        "KR1j0002", # 爾雅
        "KR1d0052", # 禮記
        "KR1e0008", # 穀梁傳
        "KR1h0004", # 論語
    ],
    "史書": [
        "KR2a0012", # 三國志
        "KR2g0017", # 列女傳
        "KR2b0003", # 前漢紀
        "KR2a0021", # 北齊書
        "KR2i0005", # 十六國春秋
        "KR2a0017", # 南齊書
        "KR2a0001", # 史記
        "KR2i0001", # 吴越春秋
        "KR2a0022", # 周書
        "KR2l0001", # 唐六典
        "KR2b0006", # 唐創業起居注
        "KR2e0001", # 國語
        "KR2a0009", # 後漢書
        "KR2b0004", # 後漢紀
        "KR2e0003", # 戰國策
        "KR2a0015", # 晉書
        "KR2d0002", # 東觀漢記
        "KR2a0018", # 梁書
        "KR2c0002", # 汲冢周書
        "KR2a0007", # 漢書
        "KR2b0001", # 竹書紀年
        "KR2i0003", # 華陽國志
        "KR2e0006", # 貞觀政要
        "KR2i0002", # 越絶書
        "KR2d0001", # 逸周書
        "KR2i0004", # 鄴中記
        "KR2a0019", # 陳書
        "KR2a0023", # 隋書
        "KR2g0018", # 高士傳
        "KR2a0020", # 魏書
    ],
    "文集": [
        "KR4b0014", # 何水部集
        "KR4b0003", # 孔北海集
        "KR4c0025", # 孟浩然集
        "KR4b0005", # 嵇中散集
        "KR4c0024", # 常建詩
        "KR4c0005", # 幽憂子集
        "KR4b0018", # 庾子山集
        "KR4b0019", # 徐孝穆集箋注
        "KR4b0001", # 揚子雲集
        "KR4i0001", # 文心雕龍
        "KR4i0004", # 文章緣起
        "KR4h0001", # 文選
        "KR4b0013", # 昭明太子集
        "KR4c0011", # 曲江集
        "KR4b0004", # 曹子建集
        "KR4c0010", # 李北海集
        "KR4c0012", # 李太白文集
        "KR4c0001", # 東臯子集
        "KR4c0004", # 楊盈川集
        "KR4a0001", # 楚辭
        "KR4b0015", # 江文通集
        "KR4h0005", # 玉臺新詠
        "KR4c0003", # 王子安集
        "KR4b0002", # 蔡中郎集
        "KR4i0003", # 詩品
        "KR4b0012", # 謝宣城詩集
        "KR4c0008", # 陳伯玉文集
        "KR4b0008", # 陶淵明集
        "KR4b0006", # 陸士衡文集
        "KR4c0021", # 須溪先生校本唐王右丞集
        "KR4c0006", # 駱丞集
        "KR4c0007", # 駱賓王文集
        "KR4c0023", # 高常侍集
        "KR4b0011", # 鮑氏集
    ],
    "諸子": [
        "KR3l0002", # 世説新語
        "KR3a0014", # 中說
        "KR3a0012", # 中論
        "KR3g0030", # 京氏易傳
        "KR3j0011", # 人物志
        "KR3a0018", # 伸蒙子
        "KR3a0013", # 傅子
        "KR3e0008", # 傷寒論
        "KR3g0003", # 元包經傳
        "KR3j0007", # 公孫龍子
        "KR3b0002", # 六韜
        "KR5c0124", # 列子
        "KR3j0013", # 劉子
        "KR3e0013", # 千金要方
        "KR3b0005", # 司馬法
        "KR3b0004", # 吳子
        "KR3j0009", # 呂氏春秋
        "KR3f0001", # 周髀算經
        "KR3g0033", # 命書
        "KR3l0006", # 唐國史補
        "KR3l0017", # 唐摭言
        "KR3l0004", # 唐新語
        "KR3l0008", # 因話錄
        "KR3j0002", # 墨子
        "KR3e0015", # 外臺祕要方
        "KR3g0024", # 天玉經
        "KR3g0001", # 太玄經
        "KR3b0011", # 太白陰經
        "KR3j0003", # 子華子
        "KR3a0003", # 孔叢子
        "KR3a0001", # 孔子家語
        "KR3b0003", # 孫子
        "KR3g0019", # 宅經
        "KR3b0006", # 尉繚子
        "KR3j0004", # 尹文子
        "KR3l0090", # 山海經
        "KR3e0012", # 巢氏諸病源候論
        "KR3a0016", # 帝範
        "KR3j0005", # 慎子
        "KR3l0098", # 拾遺記
        "KR3b0001", # 握奇經
        "KR3l0100", # 搜神後記
        "KR3l0099", # 搜神記
        "KR3g0021", # 撼龍經
        "KR3a0008", # 新序
        "KR3a0005", # 新書
        "KR3e0007", # 新編金匱要畧
        "KR3a0004", # 新語
        "KR3j0192", # 新論
        "KR3l0003", # 朝野僉載
        "KR3b0010", # 李衛公問對
        "KR3a0009", # 法言
        "KR3l0097", # 洞冥記
        "KR3l0094", # 海內十洲記
        "KR3j0010", # 淮南鴻烈
        "KR3l0096", # 漢武帝內傳
        "KR3l0095", # 漢武故事
        "KR3a0010", # 潛夫論
        "KR3g0029", # 焦氏易林
        "KR3g0034", # 玉照定真經
        "KR3e0009", # 王氏脈經
        "KR3a0011", # 申鑒
        "KR3l0101", # 異苑
        "KR3l0093", # 神異經
        "KR5a0303", # 穆天子傳
        "KR3c0001", # 管子
        "KR3a0017", # 續孟子
        "KR3l0126", # 續幽怪錄
        "KR3l0102", # 續齊諧記
        "KR5c0057", # 老子
        "KR3e0010", # 肘後備急方
        "KR3a0002", # 荀子
        "KR5c0126", # 莊子
        "KR3g0020", # 葬書
        "KR3e0011", # 褚氏遺書
        "KR3l0001", # 西京雜記
        "KR3a0007", # 說苑
        "KR3l0124", # 述異記
        "KR3l0103", # 還冤志
        "KR3l0125", # 酉陽雜俎
        "KR3e0006", # 金匱要略
        "KR3j0012", # 金樓子
        "KR3e0014", # 銀海精微
        "KR3e0005", # 鍼灸甲乙經
        "KR3g0018", # 開元占經
        "KR3g0028", # 靈棋經
        "KR3g0017", # 靈臺祕苑
        "KR3g0023", # 青囊奧語
        "KR3g0022", # 青囊序
        "KR3c0005", # 韓非子
        "KR3j0014", # 顔氏家訓
        "KR3j0001", # 鬻子
        "KR3j0008", # 鬼谷子
        "KR3j0006", # 鶡冠子
        "KR3a0006", # 鹽鐵論
        "KR3b0007", # 黃石公三略
        "KR3b0009", # 黃石公素書
        "KR3e0001", # 黄帝内經素問
        "KR3e0002", # 黄帝素問靈樞經
    ],
}

# 將 categories 轉為 repo -> category 映射
repo_to_cat = {repo: cat for cat, repos in categories.items() for repo in repos}

# 完整的 repos 列表（保持順序）
repos = [r for cat in categories.values() for r in cat]

# 目標路徑（請根據指定路徑並使用 raw string 來避免反斜線問題）
target_dir = r"D:\lufu_allusion\data\raw\compared_text"

# 確保目標資料夾存在
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# 嘗試下載每個 repository
for repo in repos:
    print(f"開始下載 {repo} ...")
    # 嘗試的分支名稱順序（先試 master，再試 main）
    branches = ["master", "main"]
    response = None
    for branch in branches:
        zip_url = f"https://github.com/kanripo/{repo}/archive/refs/heads/{branch}.zip"
        print(f"嘗試下載 {repo} 的 {branch} 分支：{zip_url}")
        try:
            response = requests.get(zip_url)
        except Exception as e:
            print(f"下載時發生錯誤：{e}")
            continue
        if response.status_code == 200:
            print(f"成功從 {branch} 分支下載 {repo}。")
            break
        else:
            print(f"{branch} 分支下載失敗，HTTP 狀態碼：{response.status_code}")

    # 如果都下載失敗則提示
    if not response or response.status_code != 200:
        print(f"下載 {repo} 失敗，請檢查 repository 是否存在或網路連線。")
        continue  # 進入下一個 repository

    # 儲存 zip 檔案的路徑
    zip_file_path = os.path.join(target_dir, f"{repo}.zip")
    
    # 儲存 zip 檔案
    with open(zip_file_path, 'wb') as f:
        f.write(response.content)
    print(f"下載完成，檔案儲存於：{zip_file_path}")
    
    # 解壓縮 zip 檔案
    try:
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(target_dir)
        print(f"解壓縮 {repo} 完成，檔案已置於：{target_dir}")
    except Exception as e:
        print(f"解壓縮 {repo} 時發生錯誤：{e}")
    
    # 刪除 zip 壓縮檔（如果不需要保留）
    try:
        os.remove(zip_file_path)
        print(f"已刪除壓縮檔：{zip_file_path}")
    except Exception as e:
        print(f"刪除壓縮檔時發生錯誤：{e}")
