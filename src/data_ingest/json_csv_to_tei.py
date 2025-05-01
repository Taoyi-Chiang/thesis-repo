# -*- coding: utf-8 -*-
# 程式：json_csv_to_tei.py
# 功能：將 JSON 列表與 CSV 詞彙註釋整合為 TEI-XML。

import json
import pandas as pd
import logging
from lxml import etree
from pathlib import Path

# 手動設定檔案路徑
TEMPLATE_XML = None  # TEI 範本 XML 檔案路徑，或 None 使用預設骨架
JSON_PATH = r"D:/lufu_allusion/sample_thesis/sample_parsed_results.json"
CSV_PATH = r"D:/lufu_allusion/sample_thesis/sample-mini-thesis.csv"
OUTPUT_XML = r"D:/lufu_allusion/sample_thesis/output.xml"

# 日誌設定
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# 命名空間設定
TEI_NS = 'http://www.tei-c.org/ns/1.0'
XML_NS = 'http://www.w3.org/XML/1998/namespace'
NSMAP = {None: TEI_NS, 'xml': XML_NS}


def create_default_template():
    # 建立最簡版 TEI 骨架，包括 xml:id 命名空間
    root = etree.Element('TEI', nsmap=NSMAP)
    etree.SubElement(root, 'teiHeader').append(etree.Element('fileDesc'))
    text_el = etree.SubElement(root, 'text')
    etree.SubElement(text_el, 'body')
    return etree.ElementTree(root)


def load_template(path):
    # 若未提供範本，回傳預設骨架；否則解析提供的 XML
    if not path:
        logging.info('使用預設 TEI 骨架')
        return create_default_template()
    parser = etree.XMLParser(remove_blank_text=True)
    return etree.parse(path, parser)


def integrate_json(tree, art):
    # art 應為 dict。將單一篇文本結構整合進 <body>
    if not isinstance(art, dict):
        logging.error('整合 JSON 時，參數必須為 dict，跳過此篇')
        return
    body = tree.find('.//text/body')
    if body is None:
        logging.error('找不到 <body>，無法整合 JSON')
        return
    # 篇 id
    fu_id = f"fu{art.get('篇號', 0)}"
    fu_div = etree.SubElement(body, 'div')
    fu_div.set('type', 'fu')
    fu_div.set(f"{{{XML_NS}}}id", fu_id)
    # 標題與作者
    head = etree.SubElement(fu_div, 'head')
    head.text = art.get('賦篇', '')
    p0 = etree.SubElement(fu_div, 'p')
    p0.text = f"作者：{art.get('賦家','')}"
    # 段落與句組
    for p in art.get('段落', []):
        pid = f"{fu_id}.p{p.get('段落編號',0)}"
        p_div = etree.SubElement(fu_div, 'div')
        p_div.set('type', 'paragraph')
        p_div.set(f"{{{XML_NS}}}id", pid)
        for g in p.get('句組', []):
            gid = f"{pid}.g{g.get('句組編號',0)}"
            lg = etree.SubElement(p_div, 'lg')
            lg.set(f"{{{XML_NS}}}id", gid)
            for s in g.get('句子', []):
                lid = f"{gid}.l{s.get('句編號',0)}"
                l = etree.SubElement(lg, 'l')
                l.set(f"{{{XML_NS}}}id", lid)
                l.text = s.get('內容','')
    logging.info(f"已整合篇號 {art.get('篇號')} 的 JSON")


def integrate_csv(tree, csv_path):
    # 讀取 CSV 並在 standOff 建立註釋
    df = pd.read_csv(csv_path, encoding='utf-8')
    root = tree.getroot()
    standOff = etree.SubElement(root, 'standOff')
    annList = etree.SubElement(standOff, 'annotationList')
    for _, row in df.iterrows():
        term = row.get('term','')
        pk   = row.get('primary key','')
        gloss= row.get('source-text','')
        src  = row.get('source','')
        ids = tree.xpath(f"//l[contains(text(),'{term}')]/@xml:id")
        if not ids:
            logging.warning(f"'{term}' 未匹配到任何句子")
            continue
        ann = etree.SubElement(annList, 'annotation')
        ann.set(f"{{{XML_NS}}}id", f"ann{pk}")
        ann.set('target', f"#{ids[0]}")
        etree.SubElement(ann, 'label').text = term
        etree.SubElement(ann, 'note', type='gloss').text = gloss
        if pd.notna(src) and src:
            etree.SubElement(ann, 'note', type='source').text = src
    logging.info('已整合 CSV 註釋')


def main():
    tree = load_template(TEMPLATE_XML)
    data = json.load(open(JSON_PATH, 'r', encoding='utf-8'))
    # 處理多篇或單篇
    if isinstance(data, list):
        for art in data:
            integrate_json(tree, art)
    else:
        integrate_json(tree, data)
    integrate_csv(tree, CSV_PATH)
    tree.write(OUTPUT_XML, encoding='utf-8', xml_declaration=True, pretty_print=True)
    logging.info(f'輸出檔案：{OUTPUT_XML}')

if __name__=='__main__':
    main()
