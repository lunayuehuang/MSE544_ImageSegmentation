import json, copy

with open('L1-dataset_prep-v5.ipynb') as f:
    l1 = json.load(f)
with open('L2-UNet-training-v5.ipynb') as f:
    l2 = json.load(f)
with open('L3-UNet-prediction-new-images-v5.ipynb') as f:
    l3 = json.load(f)

# helpers
uid_counter = [0]
def uid(tag='q'):
    uid_counter[0] += 1
    return f'{tag}-{uid_counter[0]:03d}'

def md(text):
    return {'cell_type': 'markdown', 'id': uid('md'), 'metadata': {}, 'source': [text]}

def code_clean(cell):
    c = copy.deepcopy(cell)
    c['outputs'] = []
    c['execution_count'] = None
    c.setdefault('id', uid('code'))
    return c

# ── question blocks ───────────────────────────────────────────────────────────

Q1 = md(
"---\n"
"### Question 1 — Train / Validation Split *(10 pts)*\n\n"
"Only **13 images** were labeled in this dataset, split 80/20 into train (10) and val (3).\n\n"
"1. Why do we always reserve a validation set that is **never used during training**?\n"
"2. With only 13 labeled images, what specific **overfitting risk** arises, and how does the patch-based approach partly mitigate it?\n"
"3. If you had unlimited time, what would be a more statistically robust splitting strategy than a single 80/20 split?\n\n"
"*Write your answers in a new markdown cell below this one.*"
)

Q2 = md(
"---\n"
"### Question 2 — Patch Extraction & Empty Patch Removal *(10 pts)*\n\n"
"The code tiles each source image into **256x256 patches** and discards patches where the mask is all zeros (`REMOVE_EMPTY = True`).\n\n"
"1. Why do we feed the U-Net fixed-size patches rather than variable-size full images?\n"
"2. What does `REMOVE_EMPTY = True` do to the **class balance** of the training set, and is this always a good idea?\n"
"3. If a defect spans the boundary between two adjacent patches, how might this affect model training and prediction?\n\n"
"*Write your answers in a new markdown cell below this one.*"
)

Q3 = md(
"---\n"
"### Question 3 — Data Augmentation *(10 pts)*\n\n"
"The augmentation in this baseline is intentionally **limited** to `['orig', 'fliplr']` — only the original patch and its horizontal mirror.\n\n"
"The full set of augmentations that were removed is: `'flipud'`, `'rot90'`, `'rot180'`, `'rot270'`.\n\n"
"1. Explain what each of the **4 removed augmentations** does geometrically.\n"
"2. Why are rotation and flip augmentations especially well-suited to **electron-microscopy images** of MoS2 defects?\n"
"3. Removing these augmentations reduced the training set from **~318 patches to ~106 patches**. Quantify the expected impact on model generalisation, and propose **one additional augmentation** (not in the original list) that might further help.\n\n"
"*Write your answers in a new markdown cell below this one.*"
)

Q4 = md(
"---\n"
"### Question 4 — Class Imbalance *(10 pts)*\n\n"
"The pixel statistics below show roughly **90% background** vs **10% defect** in the training masks.\n\n"
"1. What is this problem called, and why does it cause standard cross-entropy loss to produce a biased model?\n"
"2. Find **two specific mechanisms** already present in Part 2 (L2) of this notebook that address class imbalance. Name them and briefly explain how each one helps.\n"
"3. Suppose the defect class were only **1% of pixels**. Would the same mechanisms still be sufficient? What else could you do?\n\n"
"*Write your answers in a new markdown cell below this one.*"
)

Q5 = md(
"---\n"
"### Question 5 — U-Net Architecture & Skip Connections *(10 pts)*\n\n"
"The U-Net has an encoder path (d1 -> d4 -> bottleneck) and a decoder path (u4 -> u1), with skip connections at every level.\n\n"
"1. Trace the **spatial resolution** (H x W) through the encoder for a 256x256 input. At what resolution does the bottleneck operate?\n"
"2. Explain in your own words **why skip connections** are critical for pixel-wise segmentation but less commonly used in image classification networks.\n"
"3. The bottleneck `DoubleConv` uses `dropout=0.25`. What role does dropout play here, and why is the rate higher at the bottleneck than at shallower layers?\n\n"
"*Write your answers in a new markdown cell below this one.*"
)

Q6 = md(
"---\n"
"### Question 6 — Loss Function *(10 pts)*\n\n"
"The **baseline loss** in this notebook is `focal_ce_loss` only. The original also included `0.45 * focal_tversky_loss`.\n\n"
"1. Standard cross-entropy computes a per-pixel loss. What does the **focal modifier** `(1 - p_t)^gamma` do, and which pixels does it focus training on?\n"
"2. The Tversky index is: `T = TP / (TP + alpha*FP + beta*FN)`. With `alpha=0.7, beta=0.3`, which error type is penalised more (FP or FN), and is this a good choice for rare defect detection?\n"
"3. Restore the combined loss (uncomment the TODO line). What change in defect IoU do you expect and why?\n\n"
"*Write your answers in a new markdown cell below this one.*"
)

Q7 = md(
"---\n"
"### Question 7 — Training Dynamics *(10 pts)*\n\n"
"This baseline trains for only **3 epochs**. Examine the printed loss/IoU values and training curves.\n\n"
"1. Are the train and val losses still **decreasing** after epoch 3? What does this tell you about model convergence?\n"
"2. The scheduler `ReduceLROnPlateau` reduces LR when val IoU stops improving (`patience=3`). With only 3 epochs, will this scheduler ever trigger? Explain.\n"
"3. List **three concrete hyperparameter or training-loop changes** (besides restoring augmentation or fixing the loss) that would improve final defect IoU, and justify each.\n\n"
"*Write your answers in a new markdown cell below this one.*"
)

Q8 = md(
"---\n"
"### Question 8 — Evaluation & Improvement Plan *(10 pts)*\n\n"
"After training, the per-class IoU on the **validation set** is printed.\n\n"
"1. Define **Intersection over Union (IoU)** in terms of TP, FP, and FN. Why is it a better metric than pixel accuracy for this highly imbalanced task?\n"
"2. The model was trained on patches from images 13-28 but predicts on images 1-12 (Part 3). What **distribution-shift concerns** might arise, and how could you test for them?\n"
"3. Write a **concrete improvement plan** covering at least **four changes** across data, augmentation, model, and training. For each change, state the expected direction and magnitude of improvement.\n\n"
"*Write your answers in a new markdown cell below this one.*"
)

ANSWER_KEY = md(
"---\n"
"# INSTRUCTOR ANSWER KEY\n"
"*Remove or hide this section before distributing to students.*\n\n"
"---\n\n"
"## Q1 — Train / Validation Split\n\n"
"**1.** The val set is never used to update weights, providing an unbiased estimate of generalisation. Without it you cannot detect overfitting — a model can memorise training patches and achieve high train IoU while failing on unseen images.\n\n"
"**2.** With only 10 training images, image-level diversity is low; patches from the same image share lighting, magnification, and sample region. Patch-tiling gives many independent 256x256 views, but patches from the same source image remain correlated, so effective diversity is closer to 10 than 318.\n\n"
"**3.** Leave-one-image-out cross-validation (k=13) rotates every image through the val role, using all labeled data for both training and evaluation. This gives a far more reliable generalisation estimate.\n\n"
"---\n\n"
"## Q2 — Patch Extraction & Empty Patch Removal\n\n"
"**1.** U-Net downsamples 4x via max-pooling, requiring input dimensions that are multiples of 16. Fixed 256x256 patches guarantee uniform tensor shapes and allow efficient GPU batching.\n\n"
"**2.** Discarding all-zero-mask patches removes background-only tiles, making the dataset far more defect-rich and reducing within-batch class imbalance. Downside: the model sees fewer background-only examples and may produce false positives in large uniform regions.\n\n"
"**3.** A defect cut by a patch boundary appears as a partial shape in each adjacent patch. The model may underpredict near boundaries. Mitigation: use overlapping patches (stride < 256) and blend predictions with a Gaussian window.\n\n"
"---\n\n"
"## Q3 — Data Augmentation\n\n"
"**1.** Removed augmentations: `flipud` — vertical mirror (top/bottom swap); `rot90` — 90-degree counter-clockwise rotation; `rot180` — 180-degree rotation; `rot270` — 270-degree counter-clockwise rotation.\n\n"
"**2.** MoS2 crystal defects have no preferred orientation — the lattice looks statistically identical under any rotation or reflection. These transforms create physically plausible new samples without any labeling effort.\n\n"
"**3.** Reducing from 6 transforms to 2 cuts training data ~3x (318 -> ~106 patches). Expected impact: higher overfitting, lower and more variable val IoU. A useful additional augmentation: **random Gaussian noise or brightness/contrast jitter**, since EM image quality varies with beam current and focus.\n\n"
"---\n\n"
"## Q4 — Class Imbalance\n\n"
"**1.** This is called class imbalance. Standard cross-entropy is dominated by background gradients (~90% of pixels), pushing the model to predict background everywhere — achieving >90% pixel accuracy while detecting zero defects.\n\n"
"**2.** Two mechanisms in L2: (a) **Inverse-frequency class weights** (`compute_class_weights`): rarer class pixels receive higher loss weight (~6.5x for defect), amplifying their gradient contribution. (b) **WeightedRandomSampler**: patches containing defects are over-sampled (3x bonus weight), so each batch contains a higher proportion of defect patches.\n\n"
"**3.** At 1% defect these may be insufficient. Additional strategies: increase focal gamma (gamma=3.0); copy-paste augmentation (paste defect crops into background patches); two-stage approach (binary patch classifier before segmentation); hard-negative mining.\n\n"
"---\n\n"
"## Q5 — U-Net Architecture & Skip Connections\n\n"
"**1.** Resolution through encoder: input 256x256 -> d1 256x256 -> pool+d2 128x128 -> pool+d3 64x64 -> pool+d4 32x32 -> pool+bottleneck **16x16**. The bottleneck operates at 16x16 (16x spatial compression).\n\n"
"**2.** Classification only needs a global 'what' signal — spatial resolution can be discarded. Segmentation requires per-pixel 'where' predictions; high-resolution spatial detail (exact defect edges) is lost by pooling. Skip connections route this detail directly to the decoder at matching resolution, enabling sharp boundary reconstruction.\n\n"
"**3.** Dropout randomly zeroes feature maps during training, preventing co-adaptation (regularisation). The rate is highest at the bottleneck (0.25) because this layer has the highest channel count relative to spatial size (16x16x512) and bears the most overfitting risk. Shallower layers have spatial redundancy that provides implicit regularisation.\n\n"
"---\n\n"
"## Q6 — Loss Function\n\n"
"**1.** The focal modifier `(1-p_t)^gamma` down-weights confident correct predictions and concentrates gradient on hard misclassified examples. With gamma=1.5: easy example (p_t=0.9) gets weight (0.1)^1.5=0.032; hard example (p_t=0.1) gets weight (0.9)^1.5=0.854 — ~27x more gradient for hard examples.\n\n"
"**2.** With alpha=0.7 applied to FP and beta=0.3 to FN: **false positives are penalised more heavily than false negatives**. For rare defect detection, missing a defect (FN) is usually more costly than a spurious prediction. A better setting for recall would be alpha=0.3, beta=0.7. Even so, the Tversky term shifts optimisation toward region overlap rather than point-wise accuracy, which benefits sparse classes.\n\n"
"**3.** Restoring `focal_ce + 0.45*focal_tversky` typically raises defect IoU by **3-8 percentage points** because the Tversky term directly penalises incomplete defect masks (high FN).\n\n"
"---\n\n"
"## Q7 — Training Dynamics\n\n"
"**1.** After 3 epochs both losses are almost certainly still decreasing and val IoU is still trending upward — the model is far from convergence. A converged model shows a plateau in val loss and stable val IoU.\n\n"
"**2.** The scheduler needs 3 consecutive epochs without val-IoU improvement before reducing LR. Since training stops after 3 epochs it almost certainly will not trigger (val IoU is still improving at epoch 3).\n\n"
"**3.** Three changes: (a) **Increase EPOCHS to 20-30** — with batch_size=2 and ~106 patches, 3 epochs = ~159 gradient steps, far too few to converge. (b) **Cosine LR schedule** (`CosineAnnealingLR(T_max=25)`) — smoothly anneals LR, typically outperforming step-based schedules. (c) **Increase EARLY_STOPPING_PATIENCE to 7-10** — prevents premature stopping when the model briefly plateaus before resuming improvement.\n\n"
"---\n\n"
"## Q8 — Evaluation & Improvement Plan\n\n"
"**1.** IoU = TP / (TP + FP + FN). It measures the fraction of the union of predicted and true defect regions that overlap. Pixel accuracy is misleading here because predicting all-background gives >90% accuracy but IoU=0. IoU directly penalises both missed defects (FN) and spurious predictions (FP).\n\n"
"**2.** Images 1-12 may differ in beam energy, focus, or sample region from 13-28. The model may fail if contrast range, noise level, or defect morphology differs. Test: compare histogram statistics (mean, std, skewness) of patches from both sets; if significantly different, fine-tune on a small labeled subset of 1-12.\n\n"
"**3.** Improvement plan: (a) **Restore full augmentation** — ~3x more training data; expected +5-10% val IoU. (b) **Restore Focal Tversky loss** — optimises region overlap; expected +3-8% val IoU. (c) **Increase EPOCHS to 25 with cosine decay** — ensures convergence; expected +8-15% val IoU from baseline. (d) **Label additional images** (annotate images from the test set) — more labeled diversity; expected +5-15% depending on image variety. (e) **Test-time augmentation (TTA)** — average predictions over rotated/flipped versions; expected +1-3% val IoU at no training cost.\n\n"
"---\n"
"*End of answer key.*"
)

# ── assemble cells ────────────────────────────────────────────────────────────
cells = []

cells.append(md(
"# Master Notebook: L1 + L2 + L3  (Student Baseline)\n\n"
"This notebook combines dataset preparation, U-Net training, and inference into a single end-to-end workflow.\n\n"
"**The baseline is intentionally degraded** so that you can practise diagnosing and improving a real ML pipeline:\n"
"- Augmentation limited to horizontal flip only (4 rotation/flip variants removed)\n"
"- Training capped at 3 epochs\n"
"- Loss simplified to Focal Cross-Entropy only (Tversky term removed)\n\n"
"**Grading breakdown (100 pts total):**\n\n"
"| Component | Points |\n"
"|-----------|--------|\n"
"| Q1 Train/Val Split | 10 |\n"
"| Q2 Patch Extraction | 10 |\n"
"| Q3 Augmentation | 10 |\n"
"| Q4 Class Imbalance | 10 |\n"
"| Q5 U-Net Architecture | 10 |\n"
"| Q6 Loss Function | 10 |\n"
"| Q7 Training Dynamics | 10 |\n"
"| Q8 Evaluation and Improvement Plan | 10 |\n"
"| Optimised model (val defect IoU improvement) | 20 |\n"
"| **Total** | **100** |\n\n"
"> **Tip:** After answering all questions, apply your improvement plan and re-run. "
"The 20 optimisation points are awarded based on how much your final val defect IoU exceeds this baseline."
))

# ── Part 1 ────────────────────────────────────────────────────────────────────
cells.append(md('---\n# Part 1 - Dataset Preparation (L1)'))
cells.append(copy.deepcopy(l1['cells'][0]))   # L1 header markdown
cells.append(code_clean(l1['cells'][1]))       # settings + imports
cells.append(Q1)
cells.append(code_clean(l1['cells'][2]))       # output folders
cells.append(code_clean(l1['cells'][3]))       # helpers

# Augmentation -- DEGRADED
aug_src = ''.join(l1['cells'][4]['source'])
aug_degraded = aug_src.replace(
    "AUG_TAGS = ['orig', 'fliplr', 'flipud', 'rot90', 'rot180', 'rot270']",
    "# Baseline: only 2 transforms (original + horizontal flip)\n"
    "# TODO: restore full augmentation -- add 'flipud', 'rot90', 'rot180', 'rot270'\n"
    "AUG_TAGS = ['orig', 'fliplr']"
)
aug_cell = copy.deepcopy(l1['cells'][4])
aug_cell['source'] = [aug_degraded]
aug_cell['outputs'] = []
aug_cell['execution_count'] = None
cells.append(aug_cell)
cells.append(Q3)

cells.append(copy.deepcopy(l1['cells'][5]))   # "# augmentation strategy" markdown
cells.append(code_clean(l1['cells'][6]))       # collect pairs
cells.append(code_clean(l1['cells'][7]))       # train/val split
cells.append(code_clean(l1['cells'][8]))       # patch extraction
cells.append(Q2)
cells.append(copy.deepcopy(l1['cells'][9]))   # "# comments from students" markdown
# l1['cells'][10] is empty -- skip
cells.append(code_clean(l1['cells'][11]))      # YOLO yaml
cells.append(code_clean(l1['cells'][12]))      # visualisation
cells.append(code_clean(l1['cells'][13]))      # class balance stats
cells.append(Q4)
cells.append(code_clean(l1['cells'][14]))      # timestamp
cells.append(code_clean(l1['cells'][15]))      # end cell

# ── Part 2 ────────────────────────────────────────────────────────────────────
cells.append(md('---\n# Part 2 - U-Net Training (L2)'))
cells.append(code_clean(l2['cells'][0]))       # torch/CUDA check
cells.append(copy.deepcopy(l2['cells'][1]))    # L2 header markdown
cells.append(code_clean(l2['cells'][2]))       # last-edit timestamp
cells.append(code_clean(l2['cells'][3]))       # print labeled images
cells.append(code_clean(l2['cells'][4]))       # pip tqdm comment
cells.append(code_clean(l2['cells'][5]))       # pip torch comment
cells.append(copy.deepcopy(l2['cells'][6]))    # ## 1. Dataset + DataLoader
cells.append(code_clean(l2['cells'][7]))       # MoS2Dataset class
cells.append(copy.deepcopy(l2['cells'][8]))    # ## 2. U-Net Architecture
cells.append(code_clean(l2['cells'][9]))       # UNet code
cells.append(Q5)
cells.append(copy.deepcopy(l2['cells'][10]))   # ## 3. Loss

# Loss -- DEGRADED
loss_src = ''.join(l2['cells'][11]['source'])
loss_degraded = loss_src.replace(
    "def total_loss(logits, targets, class_weights):\n"
    "    return focal_ce_loss(logits, targets, class_weights) + 0.45 * focal_tversky_loss(logits, targets, class_weights)",
    "def total_loss(logits, targets, class_weights):\n"
    "    # Baseline: Focal Cross-Entropy only (Tversky term removed)\n"
    "    # TODO: restore combined loss:\n"
    "    # return focal_ce_loss(logits, targets, class_weights) + 0.45 * focal_tversky_loss(logits, targets, class_weights)\n"
    "    return focal_ce_loss(logits, targets, class_weights)"
)
loss_cell = copy.deepcopy(l2['cells'][11])
loss_cell['source'] = [loss_degraded]
loss_cell['outputs'] = []
loss_cell['execution_count'] = None
cells.append(loss_cell)
cells.append(Q6)

cells.append(copy.deepcopy(l2['cells'][12]))   # ## 4. Training
cells.append(code_clean(l2['cells'][13]))       # device selection

# Training loop -- DEGRADED
train_src = ''.join(l2['cells'][14]['source'])
train_degraded = train_src.replace(
    "BATCH_SIZE = 2 #4\nEPOCHS = 10 #30",
    "BATCH_SIZE = 2 #4\n"
    "EPOCHS = 3   # Baseline: 3 epochs -- TODO: increase to 20-30 for a fully trained model"
)
train_cell = copy.deepcopy(l2['cells'][14])
train_cell['source'] = [train_degraded]
train_cell['outputs'] = []
train_cell['execution_count'] = None
cells.append(train_cell)
cells.append(Q7)

cells.append(copy.deepcopy(l2['cells'][15]))   # ## 4b. Training History
cells.append(code_clean(l2['cells'][16]))       # history plots
cells.append(copy.deepcopy(l2['cells'][17]))   # ## 5. Visualise Predictions
cells.append(code_clean(l2['cells'][18]))       # visualise code
cells.append(code_clean(l2['cells'][19]))       # per-class IoU
cells.append(Q8)
cells.append(copy.deepcopy(l2['cells'][20]))   # last edit note

# ── Part 3 ────────────────────────────────────────────────────────────────────
cells.append(md('---\n# Part 3 - Prediction on New Images (L3)'))
for cell in l3['cells']:
    if cell['cell_type'] == 'code':
        cells.append(code_clean(cell))
    else:
        cells.append(copy.deepcopy(cell))

# ── Instructor answer key ─────────────────────────────────────────────────────
cells.append(ANSWER_KEY)

# ── write ─────────────────────────────────────────────────────────────────────
master = {
    'nbformat': 4,
    'nbformat_minor': 5,
    'metadata': l1['metadata'],
    'cells': cells
}

with open('L123-UNet-master-student-v5.ipynb', 'w') as f:
    json.dump(master, f, indent=1)

print(f'Written: L123-UNet-master-student-v5.ipynb')
print(f'Total cells: {len(cells)}')
