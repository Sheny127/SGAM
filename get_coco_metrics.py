import json
import os
import copy
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# ================= Configuration Paths =================
GT_FILE = r'D:\Lab\UAVVasteDataset\D_six\test_ground_truth1.json' 
DT_FILE = r'D:\Lab\UAVVasteDataset\D_six\predictionsgat.json'
TEST_IMG_DIR = r'D:\Lab\UAVVasteDataset\D_six\D_six\test'
# =======================================================

def main():
    # -----------------------------------------------------------
    # 1. Load Ground Truth (GT)
    # -----------------------------------------------------------
    print(f"1. Loading GT: {GT_FILE}")
    if not os.path.exists(GT_FILE):
        print("❌ Error: Ground truth (GT) file not found.")
        return
    cocoGt = COCO(GT_FILE)

    # Fix missing 'info' field
    if 'info' not in cocoGt.dataset:
        cocoGt.dataset['info'] = {'description': 'Fixed Dataset', 'version': '1.0', 'year': 2024}

    # ===========================================================
    # [Key Fix]: Force merge all categories in GT to 0 (Single-class evaluation mode)
    # ===========================================================
    print("   [Processing] Converting GT to single-class mode (All -> class 0)...")
    for ann in cocoGt.dataset['annotations']:
        ann['category_id'] = 0
    
    # Reset category definitions to a single class
    cocoGt.dataset['categories'] = [{'id': 0, 'name': 'waste'}]
    cocoGt.createIndex() # Must rebuild indices after modifying category ids
    print("   [Done] GT converted to single-class.")

    # -----------------------------------------------------------
    # 2. Load Detection Results (DT)
    # -----------------------------------------------------------
    print(f"2. Loading DT: {DT_FILE}")
    if not os.path.exists(DT_FILE):
        print("❌ Error: Detection (DT) file not found.")
        return
    with open(DT_FILE, 'r') as f:
        preds = json.load(f)

    if not preds:
        print("❌ Error: Predictions file is empty!")
        return

    # -----------------------------------------------------------
    # 3. Prepare Filename-to-ID Mapping
    # -----------------------------------------------------------
    print("3. Building mapping tables...")
    filename_to_id = {}
    for img_id, img_info in cocoGt.imgs.items():
        fname = os.path.basename(img_info['file_name']) 
        fname_no_ext = os.path.splitext(fname)[0]       
        filename_to_id[fname] = img_id          
        filename_to_id[fname_no_ext] = img_id   
    
    # -----------------------------------------------------------
    # 4. Identify Test Set Image IDs
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
    print(f"   Test set image count: {len(test_ids)}")

    if len(test_ids) == 0:
        print("❌ Warning: Test set IDs empty! Please check if TEST_IMG_DIR is correct.")
        # Fallback: Attempt evaluation with all images in GT if image directory is wrong
        print("   -> Fallback: Attempting evaluation with all images in GT...")
        test_ids = set(cocoGt.getImgIds())

    # -----------------------------------------------------------
    # 5. Rectify Predictions (DT)
    # -----------------------------------------------------------
    print("5. Matching and cleaning predictions...")
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
        
        # Keep only predictions that belong to the test set
        if real_img_id is not None and real_img_id in test_ids:
            p['image_id'] = real_img_id
            
            # [Key] Force category ID of prediction to 0 to match GT
            p['category_id'] = 0 
            new_preds.append(p)

    print(f"   Bboxes for evaluation after cleaning: {len(new_preds)} (original: {len(preds)})")
    
    if not new_preds:
        print("❌ Error: No bounding boxes matched!")
        return

    # -----------------------------------------------------------
    # 6. Evaluation
    # -----------------------------------------------------------
    print("6. Calculating metrics...")
    cocoDt = cocoGt.loadRes(new_preds)
    
    cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
    cocoEval.params.imgIds = sorted(list(test_ids))
    
    # [Key] Explicitly specify category ID [0] for evaluation
    cocoEval.params.catIds = [0]
    
    cocoEval.evaluate()
    cocoEval.accumulate()
    print("\n" + "="*60)
    print(" >>> Final Evaluation Results (Single Class: Waste) <<<")
    print("="*60)
    cocoEval.summarize()

if __name__ == '__main__':
    main()
