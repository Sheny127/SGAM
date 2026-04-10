import json
import os
import copy
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# ================= 配置路径 =================
GT_FILE = r'D:\Lab\UAVVasteDataset\D_six\test_ground_truth1.json' 
DT_FILE = r'D:\Lab\UAVVasteDataset\D_six\predictionsgat.json'
TEST_IMG_DIR = r'D:\Lab\UAVVasteDataset\D_six\D_six\test'
# ===========================================

def main():
    # -----------------------------------------------------------
    # 1. 加载 GT
    # -----------------------------------------------------------
    print(f"1. 加载 GT: {GT_FILE}")
    if not os.path.exists(GT_FILE):
        print("❌ 错误：找不到 GT 文件")
        return
    cocoGt = COCO(GT_FILE)

    # 修复 info 缺失
    if 'info' not in cocoGt.dataset:
        cocoGt.dataset['info'] = {'description': 'Fixed Dataset', 'version': '1.0', 'year': 2024}

    # ===========================================================
    # 【关键修复】: 强制将 GT 中的所有类别合并为 0 (单类别评估模式)
    # ===========================================================
    print("   [处理] 正在将 GT 转换为单类别模式 (All -> class 0)...")
    for ann in cocoGt.dataset['annotations']:
        ann['category_id'] = 0
    
    # 重置类别定义，只留一个类别
    cocoGt.dataset['categories'] = [{'id': 0, 'name': 'waste'}]
    cocoGt.createIndex() # 必须重建索引
    print("   [完成] GT 已转换为单类别。")

    # -----------------------------------------------------------
    # 2. 加载 DT
    # -----------------------------------------------------------
    print(f"2. 加载 DT: {DT_FILE}")
    if not os.path.exists(DT_FILE):
        print("❌ 错误：找不到 DT 文件")
        return
    with open(DT_FILE, 'r') as f:
        preds = json.load(f)

    if not preds:
        print("❌ 错误：预测文件为空！")
        return

    # -----------------------------------------------------------
    # 3. 准备文件名映射
    # -----------------------------------------------------------
    print("3. 建立映射表...")
    filename_to_id = {}
    for img_id, img_info in cocoGt.imgs.items():
        fname = os.path.basename(img_info['file_name']) 
        fname_no_ext = os.path.splitext(fname)[0]       
        filename_to_id[fname] = img_id          
        filename_to_id[fname_no_ext] = img_id   
    
    # -----------------------------------------------------------
    # 4. 锁定测试集图片 ID
    # -----------------------------------------------------------
    test_ids = set()
    if os.path.exists(TEST_IMG_DIR):
        for f in os.listdir(TEST_IMG_DIR):
            name = os.path.basename(f)
            name_key = os.path.splitext(name)[0]
            if name_key in filename_to_id:
                test_ids.add(filename_to_id[name_key])
            elif name in filename_to_id:
                test_ids.add(filename_to_id[name])
    print(f"   测试集图片数量: {len(test_ids)}")

    if len(test_ids) == 0:
        print("❌ 警告：测试集 ID 为空！请检查 TEST_IMG_DIR 路径是否正确。")
        # 如果路径不对，尝试使用所有 GT 中的图片进行评估
        print("   -> 尝试使用 GT 中的所有图片进行评估...")
        test_ids = set(cocoGt.getImgIds())

    # -----------------------------------------------------------
    # 5. 修正预测结果
    # -----------------------------------------------------------
    print("5. 开始匹配与清洗...")
    new_preds = []
    
    for p in preds:
        raw_id_str = str(p['image_id']) 
        
        candidates = [
            raw_id_str,              
            raw_id_str.zfill(6),
            raw_id_str.zfill(5),     
            raw_id_str + ".jpg",     
            raw_id_str.zfill(6) + ".jpg" 
        ]
        
        real_img_id = None
        for cand in candidates:
            if cand in filename_to_id:
                real_img_id = filename_to_id[cand]
                break
        
        # 只保留属于测试集的预测
        if real_img_id is not None and real_img_id in test_ids:
            p['image_id'] = real_img_id
            
            # 【关键】预测结果也强制设为 0，与 GT 对应
            p['category_id'] = 0 
            new_preds.append(p)

    print(f"   修正后用于评估的框数量: {len(new_preds)} (原始: {len(preds)})")
    
    if not new_preds:
        print("❌ 错误：没有匹配到框！")
        return

    # -----------------------------------------------------------
    # 6. 评估
    # -----------------------------------------------------------
    print("6. 开始计算指标...")
    cocoDt = cocoGt.loadRes(new_preds)
    
    cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
    cocoEval.params.imgIds = sorted(list(test_ids))
    
    # 【关键】显式指定要评估的类别 ID 为 [0]
    cocoEval.params.catIds = [0]
    
    cocoEval.evaluate()
    cocoEval.accumulate()
    print("\n" + "="*60)
    print(" >>> 最终测试结果 (Single Class: Waste) <<<")
    print("="*60)
    cocoEval.summarize()

if __name__ == '__main__':
    main()