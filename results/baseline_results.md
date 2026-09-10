# Next POI Recommendation Baseline Results

- Generated: 2026-09-10 15:07:05
- Setting: **Next POI recommendation** on Foursquare **NYC** / **Tokyo (TKY)**
- Data: GETNext-style processed splits under `datasets/processed/{NYC,TKY}/`
- Metrics: Acc@1 / Acc@5 / Acc@10 / Recall@5 / Recall@10 / MRR（单 GT 时 Acc@k = Recall@k）
- Framework: PyTorch；结果来自 `results/<MODEL>/{NYC,TKY}.json`

## Summary Tables (Test)

### NYC

| Model | Acc@1 | Acc@5 | Acc@10 | Recall@5 | Recall@10 | MRR | #Test |
|---|---:|---:|---:|---:|---:|---:|---:|
| FPMC | 0.1523 | 0.4142 | 0.5059 | 0.4142 | 0.5059 | 0.2685 | 1451 |
| PLSPL | 0.2026 | 0.4252 | 0.5003 | 0.4252 | 0.5003 | 0.3026 | 1451 |
| STGCN | 0.1964 | 0.4459 | 0.5424 | 0.4459 | 0.5424 | 0.3078 | 1451 |
| STGN | 0.2074 | 0.4542 | 0.5403 | 0.4542 | 0.5403 | 0.3171 | 1451 |
| ST-RNN | 0.1971 | 0.4528 | 0.5479 | 0.4528 | 0.5479 | 0.3119 | 1451 |
| GETNext | 0.2147 | 0.4574 | 0.5507 | 0.4574 | 0.5507 | 0.3264 | 1351 |
| STAN | 0.0917 | 0.2288 | 0.2813 | 0.2288 | 0.2813 | 0.1547 | 1429 |
| MTNet | 0.2047 | 0.4528 | 0.5417 | 0.4528 | 0.5417 | 0.3149 | 1451 |
| DCHL | 0.2081 | 0.4452 | 0.5410 | 0.4452 | 0.5410 | 0.3149 | 1451 |
| iPCM | 0.2212 | 0.4776 | 0.5679 | 0.4776 | 0.5679 | 0.3383 | 1451 |
| K1-POI | 0.1902 | 0.4287 | 0.5479 | 0.4287 | 0.5479 | 0.3036 | 1451 |
| STHGCN | 0.1895 | 0.4349 | 0.5472 | 0.4349 | 0.5472 | 0.3035 | 1451 |

### TKY

| Model | Acc@1 | Acc@5 | Acc@10 | Recall@5 | Recall@10 | MRR | #Test |
|---|---:|---:|---:|---:|---:|---:|---:|
| FPMC | 0.0921 | 0.2566 | 0.3651 | 0.2566 | 0.3651 | 0.1781 | 4766 |
| PLSPL | 0.2390 | 0.4425 | 0.5206 | 0.4425 | 0.5206 | 0.3332 | 4766 |
| STGCN | 0.2480 | 0.4719 | 0.5569 | 0.4719 | 0.5569 | 0.3511 | 4766 |
| STGN | 0.2411 | 0.4727 | 0.5575 | 0.4727 | 0.5575 | 0.3481 | 4766 |
| ST-RNN | 0.2384 | 0.4627 | 0.5522 | 0.4627 | 0.5522 | 0.3408 | 4766 |
| GETNext | 0.1469 | 0.2822 | 0.3282 | 0.2822 | 0.3282 | 0.2112 | 4738 |
| STAN | 0.0017 | 0.0052 | 0.0101 | 0.0052 | 0.0101 | 0.0052 | 4762 |
| MTNet | 0.2648 | 0.4872 | 0.5676 | 0.4872 | 0.5676 | 0.3676 | 4766 |
| DCHL | 0.2562 | 0.4748 | 0.5638 | 0.4748 | 0.5638 | 0.3594 | 4766 |
| iPCM | 0.2581 | 0.4987 | 0.5909 | 0.4987 | 0.5909 | 0.3684 | 4766 |
| K1-POI | 0.2528 | 0.4887 | 0.5804 | 0.4887 | 0.5804 | 0.3625 | 4766 |
| STHGCN | 0.2554 | 0.4792 | 0.5650 | 0.4792 | 0.5650 | 0.3593 | 4766 |

## Ranked by Acc@1

### NYC

| Rank | Model | Acc@1 | Acc@5 | Acc@10 | MRR |
|---:|---|---:|---:|---:|---:|
| 1 | iPCM | 0.2212 | 0.4776 | 0.5679 | 0.3383 |
| 2 | GETNext | 0.2147 | 0.4574 | 0.5507 | 0.3264 |
| 3 | DCHL | 0.2081 | 0.4452 | 0.5410 | 0.3149 |
| 4 | STGN | 0.2074 | 0.4542 | 0.5403 | 0.3171 |
| 5 | MTNet | 0.2047 | 0.4528 | 0.5417 | 0.3149 |
| 6 | PLSPL | 0.2026 | 0.4252 | 0.5003 | 0.3026 |
| 7 | ST-RNN | 0.1971 | 0.4528 | 0.5479 | 0.3119 |
| 8 | STGCN | 0.1964 | 0.4459 | 0.5424 | 0.3078 |
| 9 | K1-POI | 0.1902 | 0.4287 | 0.5479 | 0.3036 |
| 10 | STHGCN | 0.1895 | 0.4349 | 0.5472 | 0.3035 |
| 11 | FPMC | 0.1523 | 0.4142 | 0.5059 | 0.2685 |
| 12 | STAN | 0.0917 | 0.2288 | 0.2813 | 0.1547 |

### TKY

| Rank | Model | Acc@1 | Acc@5 | Acc@10 | MRR |
|---:|---|---:|---:|---:|---:|
| 1 | MTNet | 0.2648 | 0.4872 | 0.5676 | 0.3676 |
| 2 | iPCM | 0.2581 | 0.4987 | 0.5909 | 0.3684 |
| 3 | DCHL | 0.2562 | 0.4748 | 0.5638 | 0.3594 |
| 4 | STHGCN | 0.2554 | 0.4792 | 0.5650 | 0.3593 |
| 5 | K1-POI | 0.2528 | 0.4887 | 0.5804 | 0.3625 |
| 6 | STGCN | 0.2480 | 0.4719 | 0.5569 | 0.3511 |
| 7 | STGN | 0.2411 | 0.4727 | 0.5575 | 0.3481 |
| 8 | PLSPL | 0.2390 | 0.4425 | 0.5206 | 0.3332 |
| 9 | ST-RNN | 0.2384 | 0.4627 | 0.5522 | 0.3408 |
| 10 | GETNext | 0.1469 | 0.2822 | 0.3282 | 0.2112 |
| 11 | FPMC | 0.0921 | 0.2566 | 0.3651 | 0.1781 |
| 12 | STAN | 0.0017 | 0.0052 | 0.0101 | 0.0052 |

## Notes / Caveats

- **GETNext-TKY**：官方 forward 复现后仍在训练中后期数值爆炸；当前表内数字来自 early-stop 前 best checkpoint，显著弱于 NYC，解读需谨慎。
- **STAN-TKY**：结果文件存在，但 Acc@1≈0.002，基本未学到有效排序，建议后续单独排查负采样/评估协议。
- **样本数差异**：部分模型 test `n` 为 1351/1451（NYC）或 4738/4766（TKY），来自不同过滤（如是否要求 user 出现在 train）。横向对比时注意。
- GETNext NYC 使用 `align=official-forward`；其余模型走统一 Next-POI 评估协议（last-timestep full-candidate ranking）。

## Per-run Details

| Model | City | Epochs recorded | Device | Result file | Last updated |
|---|---|---:|---|---|---|
| FPMC | NYC | 21 | cuda | `results/FPMC/NYC.json` | 2026-09-07 23:53:45 |
| FPMC | TKY | 18 | cuda | `results/FPMC/TKY.json` | 2026-09-07 23:56:11 |
| PLSPL | NYC | 30 | cuda | `results/PLSPL/NYC.json` | 2026-09-07 23:56:02 |
| PLSPL | TKY | 30 | cuda | `results/PLSPL/TKY.json` | 2026-09-08 00:04:05 |
| STGCN | NYC | 26 | cuda | `results/STGCN/NYC.json` | 2026-09-08 00:10:36 |
| STGCN | TKY | 30 | cuda | `results/STGCN/TKY.json` | 2026-09-08 00:58:39 |
| STGN | NYC | 30 | cuda | `results/STGN/NYC.json` | 2026-09-08 00:26:43 |
| STGN | TKY | 30 | cuda | `results/STGN/TKY.json` | 2026-09-08 01:19:29 |
| ST-RNN | NYC | 30 | cuda | `results/ST-RNN/NYC.json` | 2026-09-08 01:09:14 |
| ST-RNN | TKY | 30 | cuda | `results/ST-RNN/TKY.json` | 2026-09-08 01:44:49 |
| GETNext | NYC | 44 | cuda | `results/GETNext/NYC.json` | 2026-09-07 18:39:09 |
| GETNext | TKY | 22 | cuda | `results/GETNext/TKY.json` | 2026-09-08 13:22:07 |
| STAN | NYC | 27 | cuda | `results/STAN/NYC.json` | 2026-09-07 21:47:03 |
| STAN | TKY | 9 | cuda | `results/STAN/TKY.json` | 2026-09-08 21:40:08 |
| MTNet | NYC | 17 | - | `results/MTNet/NYC.json` | 2026-09-09 15:37:09 |
| MTNet | TKY | 26 | - | `results/MTNet/TKY.json` | 2026-09-09 15:44:58 |
| DCHL | NYC | 16 | - | `results/DCHL/NYC.json` | 2026-09-09 15:41:09 |
| DCHL | TKY | 30 | - | `results/DCHL/TKY.json` | 2026-09-09 16:04:53 |
| iPCM | NYC | 19 | - | `results/iPCM/NYC.json` | 2026-09-09 15:47:50 |
| iPCM | TKY | 15 | - | `results/iPCM/TKY.json` | 2026-09-09 15:55:56 |
| K1-POI | NYC | 10 | - | `results/K1-POI/NYC.json` | 2026-09-09 15:57:05 |
| K1-POI | TKY | 11 | - | `results/K1-POI/TKY.json` | 2026-09-09 16:01:00 |
| STHGCN | NYC | 12 | - | `results/STHGCN/NYC.json` | 2026-09-09 16:03:57 |
| STHGCN | TKY | 20 | - | `results/STHGCN/TKY.json` | 2026-09-09 16:17:47 |

## Known Issue Notes

- **GETNext-TKY**: 训练中后期出现数值不稳定（大量 batch 被 skip）；TEST 取 early best（约 epoch 2）。结果可信度低于 NYC。
- **STAN-TKY**: 已跑完并写出结果，但指标接近随机，复现质量较差，不宜作为强结论。
- **STAN-NYC**: 已跑通，但相对其他 baseline 偏低。

## Source Layout

```
results/
  FPMC/
    NYC.json
    TKY.json
  PLSPL/
    NYC.json
    TKY.json
  STGCN/
    NYC.json
    TKY.json
  STGN/
    NYC.json
    TKY.json
  ST-RNN/
    NYC.json
    TKY.json
  GETNext/
    NYC.json
    TKY.json
  STAN/
    NYC.json
    TKY.json
  MTNet/
    NYC.json
    TKY.json
  DCHL/
    NYC.json
    TKY.json
  iPCM/
    NYC.json
    TKY.json
  K1-POI/
    NYC.json
    TKY.json
  STHGCN/
    NYC.json
    TKY.json
```

Checkpoints are stored under `checkpoints/<MODEL>/...`.
