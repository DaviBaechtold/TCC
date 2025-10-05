_base_ = ['./rtmpose_m_wholebody.py']

# Override validation to use training annotations and images
val_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        ann_file='annotations/coco_wholebody_train_v1.0.json',
        data_prefix=dict(img='train2017/'),
        test_mode=True,
    ))

test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoWholeBodyMetric',
    ann_file='data/processed/grayscale/annotations/coco_wholebody_train_v1.0.json')

test_evaluator = val_evaluator

# Ensure val/test cfg present when providing evaluators and dataloaders
val_cfg = dict()
test_cfg = dict()
