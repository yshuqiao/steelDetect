# optimizer
# optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer = dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001)
# optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.00005)
optimizer_config = dict(grad_clip=None)
# optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',  # 可改为余弦退火算法
    warmup_iters=500,  # 在初始的500次迭代中学习率逐渐增加
    warmup_ratio=0.001,  # 起始的学习率
    step=[8, 11]  # 在第8和11个epoch时降低学习率，可调成step=[16, 19]
# step=[10, 14]
# step=[16, 22]
)

# lr_config = dict(
#     policy='CosineAnnealing',
#     warmup='linear',
#     warmup_iters=1000,
#     warmup_ratio=1.0/10,
#     min_lr_ratio=1e-5,
# )



total_epochs = 12  # 可调成20
# total_epochs = 16
# total_epochs = 24