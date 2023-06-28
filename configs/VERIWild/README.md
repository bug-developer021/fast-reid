# Model architecture

## veriwild
```txt
[Config]: 
 CUDNN_BENCHMARK: True
DATALOADER:
  NAIVE_WAY: True
  NUM_INSTANCE: 4
  NUM_WORKERS: 8
  PK_SAMPLER: True
DATASETS:
  COMBINEALL: False
  NAMES: ('VeRiWild',)
  TESTS: ('SmallVeRiWild', 'MediumVeRiWild', 'LargeVeRiWild')
INPUT:
  AUGMIX_PROB: 0.0
  AUTOAUG_PROB: 0.0
  CJ:
    BRIGHTNESS: 0.15
    CONTRAST: 0.15
    ENABLED: False
    HUE: 0.1
    PROB: 0.5
    SATURATION: 0.1
  DO_AFFINE: False
  DO_AUGMIX: False
  DO_AUTOAUG: False
  DO_FLIP: True
  DO_PAD: True
  FLIP_PROB: 0.5
  PADDING: 10
  PADDING_MODE: constant
  REA:
    ENABLED: True
    PROB: 0.5
    VALUE: [123.675, 116.28, 103.53]
  RPT:
    ENABLED: False
    PROB: 0.5
  SIZE_TEST: [256, 256]
  SIZE_TRAIN: [256, 256]
KD:
  MODEL_CONFIG: ['']
  MODEL_WEIGHTS: ['']
MODEL:
  BACKBONE:
    DEPTH: 50x
    FEAT_DIM: 2048
    LAST_STRIDE: 1
    NAME: build_resnet_backbone
    NORM: BN
    PRETRAIN: False
    PRETRAIN_PATH: 
    WITH_IBN: True
    WITH_NL: False
    WITH_SE: False
  DEVICE: cuda:0
  FREEZE_LAYERS: ['']
  HEADS:
    CLS_LAYER: linear
    EMBEDDING_DIM: 0
    MARGIN: 0.15
    NAME: EmbeddingHead
    NECK_FEAT: before
    NORM: BN
    NUM_CLASSES: 0
    POOL_LAYER: gempool
    SCALE: 128
    WITH_BNNECK: True
  LOSSES:
    CE:
      ALPHA: 0.2
      EPSILON: 0.1
      SCALE: 1.0
    CIRCLE:
      GAMMA: 128
      MARGIN: 0.25
      SCALE: 1.0
    COSFACE:
      GAMMA: 128
      MARGIN: 0.25
      SCALE: 1.0
    FL:
      ALPHA: 0.25
      GAMMA: 2
      SCALE: 1.0
    NAME: ('CrossEntropyLoss', 'TripletLoss')
    TRI:
      HARD_MINING: False
      MARGIN: 0.0
      NORM_FEAT: False
      SCALE: 1.0
  META_ARCHITECTURE: Baseline
  PIXEL_MEAN: [123.675, 116.28, 103.53]
  PIXEL_STD: [58.395, 57.120000000000005, 57.375]
  QUEUE_SIZE: 8192
  WEIGHTS: ../Yolov8-tracking-alago/weights/veriwild_bot_R50-ibn.pth
OUTPUT_DIR: logs/veriwild/bagtricks_R50-ibn_4gpu
SOLVER:
  BASE_LR: 0.00035
  BIAS_LR_FACTOR: 2.0
  CHECKPOINT_PERIOD: 20
  DELAY_EPOCHS: 0
  ETA_MIN_LR: 1e-07
  FP16_ENABLED: True
  FREEZE_FC_ITERS: 0
  FREEZE_ITERS: 0
  GAMMA: 0.1
  HEADS_LR_FACTOR: 1.0
  IMS_PER_BATCH: 512
  MAX_EPOCH: 120
  MOMENTUM: 0.9
  NESTEROV: True
  OPT: Adam
  SCHED: MultiStepLR
  STEPS: [30, 70, 90]
  WARMUP_FACTOR: 0.1
  WARMUP_ITERS: 5000
  WARMUP_METHOD: linear
  WEIGHT_DECAY: 0.0005
  WEIGHT_DECAY_BIAS: 0.0005
TEST:
  AQE:
    ALPHA: 3.0
    ENABLED: False
    QE_K: 5
    QE_TIME: 1
  EVAL_PERIOD: 10
  FLIP_ENABLED: False
  IMS_PER_BATCH: 128
  METRIC: cosine
  PRECISE_BN:
    DATASET: Market1501
    ENABLED: False
    NUM_ITER: 300
  RERANK:
    ENABLED: False
    K1: 20
    K2: 6
    LAMBDA: 0.3
  ROC_ENABLED: False
/usr/local/lib/python3.8/dist-packages/torch/nn/init.py:405: UserWarning: Initializing zero-element tensors is a no-op
  warnings.warn("Initializing zero-element tensors is a no-op")
[Model]: 
 Baseline(
  (backbone): ResNet(
    (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    (bn1): BatchNorm(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (relu): ReLU(inplace=True)
    (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True)
    (layer1): Sequential(
      (0): Bottleneck(
        (conv1): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): IBN(
          (IN): InstanceNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (BN): BatchNorm(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (se): Identity()
        (downsample): Sequential(
          (0): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (1): Bottleneck(
        (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): IBN(
          (IN): InstanceNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (BN): BatchNorm(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (se): Identity()
      )
      (2): Bottleneck(
        (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): IBN(
          (IN): InstanceNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (BN): BatchNorm(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (se): Identity()
      )
    )
    (layer2): Sequential(
      (0): Bottleneck(
        (conv1): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): IBN(
          (IN): InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (BN): BatchNorm(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (bn2): BatchNorm(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (se): Identity()
        (downsample): Sequential(
          (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
          (1): BatchNorm(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (1): Bottleneck(
        (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): IBN(
          (IN): InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (BN): BatchNorm(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (se): Identity()
      )
      (2): Bottleneck(
        (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): IBN(
          (IN): InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (BN): BatchNorm(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (se): Identity()
      )
      (3): Bottleneck(
        (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): IBN(
          (IN): InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (BN): BatchNorm(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (se): Identity()
      )
    )
    (layer3): Sequential(
      (0): Bottleneck(
        (conv1): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): IBN(
          (IN): InstanceNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (BN): BatchNorm(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (bn2): BatchNorm(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (se): Identity()
        (downsample): Sequential(
          (0): Conv2d(512, 1024, kernel_size=(1, 1), stride=(2, 2), bias=False)
          (1): BatchNorm(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (1): Bottleneck(
        (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): IBN(
          (IN): InstanceNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (BN): BatchNorm(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (se): Identity()
      )
      (2): Bottleneck(
        (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): IBN(
          (IN): InstanceNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (BN): BatchNorm(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (se): Identity()
      )
      (3): Bottleneck(
        (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): IBN(
          (IN): InstanceNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (BN): BatchNorm(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (se): Identity()
      )
      (4): Bottleneck(
        (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): IBN(
          (IN): InstanceNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (BN): BatchNorm(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (se): Identity()
      )
      (5): Bottleneck(
        (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): IBN(
          (IN): InstanceNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (BN): BatchNorm(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (se): Identity()
      )
    )
    (layer4): Sequential(
      (0): Bottleneck(
        (conv1): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNorm(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (se): Identity()
        (downsample): Sequential(
          (0): Conv2d(1024, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (1): Bottleneck(
        (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNorm(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (se): Identity()
      )
      (2): Bottleneck(
        (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNorm(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (se): Identity()
      )
    )
  )
  (heads): EmbeddingHead(
    (pool_layer): GeneralizedMeanPooling(3.0, output_size=1)
    (bottleneck): Sequential(
      (0): BatchNorm(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    (classifier): Linear(in_features=2048, out_features=0, bias=False)
  )
)
```

## veri

```
[Config]: 
 CUDNN_BENCHMARK: True
DATALOADER:
  NAIVE_WAY: True
  NUM_INSTANCE: 16
  NUM_WORKERS: 8
  PK_SAMPLER: True
DATASETS:
  COMBINEALL: False
  NAMES: ('VeRi',)
  TESTS: ('VeRi',)
INPUT:
  AUGMIX_PROB: 0.0
  AUTOAUG_PROB: 0.1
  CJ:
    BRIGHTNESS: 0.15
    CONTRAST: 0.15
    ENABLED: False
    HUE: 0.1
    PROB: 0.5
    SATURATION: 0.1
  DO_AFFINE: False
  DO_AUGMIX: False
  DO_AUTOAUG: True
  DO_FLIP: True
  DO_PAD: True
  FLIP_PROB: 0.5
  PADDING: 10
  PADDING_MODE: constant
  REA:
    ENABLED: True
    PROB: 0.5
    VALUE: [123.675, 116.28, 103.53]
  RPT:
    ENABLED: False
    PROB: 0.5
  SIZE_TEST: [256, 256]
  SIZE_TRAIN: [256, 256]
KD:
  MODEL_CONFIG: ['']
  MODEL_WEIGHTS: ['']
MODEL:
  BACKBONE:
    DEPTH: 50x
    FEAT_DIM: 2048
    LAST_STRIDE: 1
    NAME: build_resnet_backbone
    NORM: BN
    PRETRAIN: False
    PRETRAIN_PATH: 
    WITH_IBN: True
    WITH_NL: True
    WITH_SE: False
  DEVICE: cuda:0
  FREEZE_LAYERS: ['backbone']
  HEADS:
    CLS_LAYER: circleSoftmax
    EMBEDDING_DIM: 0
    MARGIN: 0.35
    NAME: EmbeddingHead
    NECK_FEAT: after
    NORM: BN
    NUM_CLASSES: 0
    POOL_LAYER: gempoolP
    SCALE: 64
    WITH_BNNECK: True
  LOSSES:
    CE:
      ALPHA: 0.2
      EPSILON: 0.1
      SCALE: 1.0
    CIRCLE:
      GAMMA: 128
      MARGIN: 0.25
      SCALE: 1.0
    COSFACE:
      GAMMA: 128
      MARGIN: 0.25
      SCALE: 1.0
    FL:
      ALPHA: 0.25
      GAMMA: 2
      SCALE: 1.0
    NAME: ('CrossEntropyLoss', 'TripletLoss')
    TRI:
      HARD_MINING: True
      MARGIN: 0.0
      NORM_FEAT: False
      SCALE: 1.0
  META_ARCHITECTURE: Baseline
  PIXEL_MEAN: [123.675, 116.28, 103.53]
  PIXEL_STD: [58.395, 57.120000000000005, 57.375]
  QUEUE_SIZE: 8192
  WEIGHTS: ./veri_sbs_R50-ibn.pth
OUTPUT_DIR: logs/veri/sbs_R50-ibn
SOLVER:
  BASE_LR: 0.01
  BIAS_LR_FACTOR: 1.0
  CHECKPOINT_PERIOD: 20
  DELAY_EPOCHS: 30
  ETA_MIN_LR: 7.7e-05
  FP16_ENABLED: False
  FREEZE_FC_ITERS: 0
  FREEZE_ITERS: 1000
  GAMMA: 0.1
  HEADS_LR_FACTOR: 1.0
  IMS_PER_BATCH: 64
  MAX_EPOCH: 60
  MOMENTUM: 0.9
  NESTEROV: True
  OPT: SGD
  SCHED: CosineAnnealingLR
  STEPS: [40, 90]
  WARMUP_FACTOR: 0.1
  WARMUP_ITERS: 2000
  WARMUP_METHOD: linear
  WEIGHT_DECAY: 0.0005
  WEIGHT_DECAY_BIAS: 0.0005
TEST:
  AQE:
    ALPHA: 3.0
    ENABLED: False
    QE_K: 5
    QE_TIME: 1
  EVAL_PERIOD: 20
  FLIP_ENABLED: False
  IMS_PER_BATCH: 128
  METRIC: cosine
  PRECISE_BN:
    DATASET: Market1501
    ENABLED: False
    NUM_ITER: 300
  RERANK:
    ENABLED: False
    K1: 20
    K2: 6
    LAMBDA: 0.3
  ROC_ENABLED: False
/usr/local/lib/python3.8/dist-packages/torch/nn/init.py:405: UserWarning: Initializing zero-element tensors is a no-op
  warnings.warn("Initializing zero-element tensors is a no-op")
[Model]: 
 Baseline(
  (backbone): ResNet(
    (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    (bn1): BatchNorm(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (relu): ReLU(inplace=True)
    (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True)
    (layer1): Sequential(
      (0): Bottleneck(
        (conv1): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): IBN(
          (IN): InstanceNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (BN): BatchNorm(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (se): Identity()
        (downsample): Sequential(
          (0): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (1): Bottleneck(
        (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): IBN(
          (IN): InstanceNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (BN): BatchNorm(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (se): Identity()
      )
      (2): Bottleneck(
        (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): IBN(
          (IN): InstanceNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (BN): BatchNorm(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (se): Identity()
      )
    )
    (layer2): Sequential(
      (0): Bottleneck(
        (conv1): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): IBN(
          (IN): InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (BN): BatchNorm(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (bn2): BatchNorm(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (se): Identity()
        (downsample): Sequential(
          (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
          (1): BatchNorm(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (1): Bottleneck(
        (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): IBN(
          (IN): InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (BN): BatchNorm(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (se): Identity()
      )
      (2): Bottleneck(
        (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): IBN(
          (IN): InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (BN): BatchNorm(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (se): Identity()
      )
      (3): Bottleneck(
        (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): IBN(
          (IN): InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (BN): BatchNorm(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (se): Identity()
      )
    )
    (layer3): Sequential(
      (0): Bottleneck(
        (conv1): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): IBN(
          (IN): InstanceNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (BN): BatchNorm(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (bn2): BatchNorm(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (se): Identity()
        (downsample): Sequential(
          (0): Conv2d(512, 1024, kernel_size=(1, 1), stride=(2, 2), bias=False)
          (1): BatchNorm(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (1): Bottleneck(
        (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): IBN(
          (IN): InstanceNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (BN): BatchNorm(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (se): Identity()
      )
      (2): Bottleneck(
        (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): IBN(
          (IN): InstanceNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (BN): BatchNorm(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (se): Identity()
      )
      (3): Bottleneck(
        (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): IBN(
          (IN): InstanceNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (BN): BatchNorm(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (se): Identity()
      )
      (4): Bottleneck(
        (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): IBN(
          (IN): InstanceNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (BN): BatchNorm(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (se): Identity()
      )
      (5): Bottleneck(
        (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): IBN(
          (IN): InstanceNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (BN): BatchNorm(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (se): Identity()
      )
    )
    (layer4): Sequential(
      (0): Bottleneck(
        (conv1): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNorm(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (se): Identity()
        (downsample): Sequential(
          (0): Conv2d(1024, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (1): Bottleneck(
        (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNorm(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (se): Identity()
      )
      (2): Bottleneck(
        (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNorm(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (se): Identity()
      )
    )
    (NL_1): ModuleList()
    (NL_2): ModuleList(
      (0): Non_local(
        (g): Conv2d(512, 1, kernel_size=(1, 1), stride=(1, 1))
        (W): Sequential(
          (0): Conv2d(1, 512, kernel_size=(1, 1), stride=(1, 1))
          (1): BatchNorm(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (theta): Conv2d(512, 1, kernel_size=(1, 1), stride=(1, 1))
        (phi): Conv2d(512, 1, kernel_size=(1, 1), stride=(1, 1))
      )
      (1): Non_local(
        (g): Conv2d(512, 1, kernel_size=(1, 1), stride=(1, 1))
        (W): Sequential(
          (0): Conv2d(1, 512, kernel_size=(1, 1), stride=(1, 1))
          (1): BatchNorm(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (theta): Conv2d(512, 1, kernel_size=(1, 1), stride=(1, 1))
        (phi): Conv2d(512, 1, kernel_size=(1, 1), stride=(1, 1))
      )
    )
    (NL_3): ModuleList(
      (0): Non_local(
        (g): Conv2d(1024, 1, kernel_size=(1, 1), stride=(1, 1))
        (W): Sequential(
          (0): Conv2d(1, 1024, kernel_size=(1, 1), stride=(1, 1))
          (1): BatchNorm(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (theta): Conv2d(1024, 1, kernel_size=(1, 1), stride=(1, 1))
        (phi): Conv2d(1024, 1, kernel_size=(1, 1), stride=(1, 1))
      )
      (1): Non_local(
        (g): Conv2d(1024, 1, kernel_size=(1, 1), stride=(1, 1))
        (W): Sequential(
          (0): Conv2d(1, 1024, kernel_size=(1, 1), stride=(1, 1))
          (1): BatchNorm(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (theta): Conv2d(1024, 1, kernel_size=(1, 1), stride=(1, 1))
        (phi): Conv2d(1024, 1, kernel_size=(1, 1), stride=(1, 1))
      )
      (2): Non_local(
        (g): Conv2d(1024, 1, kernel_size=(1, 1), stride=(1, 1))
        (W): Sequential(
          (0): Conv2d(1, 1024, kernel_size=(1, 1), stride=(1, 1))
          (1): BatchNorm(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (theta): Conv2d(1024, 1, kernel_size=(1, 1), stride=(1, 1))
        (phi): Conv2d(1024, 1, kernel_size=(1, 1), stride=(1, 1))
      )
    )
    (NL_4): ModuleList()
  )
  (heads): EmbeddingHead(
    (pool_layer): GeneralizedMeanPoolingP(Parameter containing:
    tensor([3.], device='cuda:0', requires_grad=True), output_size=1)
    (bottleneck): Sequential(
      (0): BatchNorm(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    (classifier): CircleSoftmax(in_features=2048, num_classes=0, scale=64, margin=0.35)
  )
)
```