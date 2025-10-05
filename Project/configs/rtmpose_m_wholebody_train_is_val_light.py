_base_ = ['./rtmpose_m_wholebody_train_is_val.py']

# Smaller batch/workers to reduce OOM risk on first run
train_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
)

val_dataloader = dict(
    batch_size=16,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
)

# Log with shorter window to see early progress
log_processor = dict(
    type='LogProcessor',
    window_size=20,
    by_epoch=True,
    num_digits=6)
