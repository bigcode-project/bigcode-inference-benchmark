# bigcode-inference-benchmark
A100 80GB

BLOOM
```python
hidden_size = 2048
n_head = 16
n_layer = 24
total_params = 1311535104
```

Throughput (tokens/sec | msec/token)
| batch_size |    HF (fp32)    |    HF (bf16)    |    HF (int8)    | DS-inference (fp16) |
|:----------:|:---------------:|:---------------:|:---------------:|:-------------------:|
| 1          | 51.59 \| 19.38  | 47.46 \| 21.07  | 16.53 \| 60.49  | 61.61 \| 16.23      |
| 2          | 103.92 \| 9.62  | 96.88 \| 10.32  | 33.79 \| 29.60  | 121.55 \| 8.23      |
| 4          | 211.96 \| 4.72  | 193.72 \| 5.16  | 67.38 \| 14.84  | 240.06 \| 4.17      |
| 8          | 411.79 \| 2.43  | 370.67 \| 2.70  | 134.34 \| 7.44  | 492.42 \| 2.03      |
| 16         | 804.55 \| 1.24  | 781.29 \| 1.28  | 275.69 \| 3.63  | 970.59 \| 1.03      |
| 32         | 1574.68 \| 0.64 | 1539.19 \| 0.65 | 537.14 \| 1.86  | 1999.04 \| 0.50     |
| 64         | 2712.46 \| 0.37 | 3038.01 \| 0.33 | 1070.50 \| 0.93 | 3971.09 \| 0.25     |
| 128        | 2974.36 \| 0.34 | 5795.97 \| 0.17 | 2055.34 \| 0.49 | 7514.59 \| 0.13     |
| 256        | 3695.44 \| 0.27 | 8216.27 \| 0.12 | 3523.77 \| 0.28 | 10226.50 \| 0.10    |
| 384        | 3591.13 \| 0.28 | 9328.18 \| 0.11 | 4585.33 \| 0.22 | 11094.27 \| 0.09    |
| 512        | 3708.54 \| 0.27 | 9446.34 \| 0.11 | 5416.48 \| 0.18 | 11390.85 \| 0.09    |
| 640        | 3859.43 \| 0.26 | 9572.53 \| 0.10 | 6113.65 \| 0.16 | 11625.71 \| 0.09    |
| 768        | 3804.82 \| 0.26 | 9464.75 \| 0.11 | 6582.52 \| 0.15 | 11814.31 \| 0.08    |
| 896        | 3652.42 \| 0.27 | 9482.11 \| 0.11 | 7111.08 \| 0.14 | 11744.38 \| 0.09    |
| 1024       | oom             | 9710.46 \| 0.10 | 7486.36 \| 0.13 | 11534.95 \| 0.09    |
| 1152       | oom             | 9712.39 \| 0.10 | 7544.99 \| 0.13 | oom                 |
| 1280       | oom             | 9667.19 \| 0.10 | 7858.91 \| 0.13 | oom                 |
| 1408       | oom             | 9771.91 \| 0.10 | 8116.30 \| 0.12 | oom                 |
| 1536       | oom             | 9744.56 \| 0.10 | 8201.28 \| 0.12 | oom                 |
| 1664       | oom             | 9719.82 \| 0.10 | 8227.56 \| 0.12 | oom                 |
| 1792       | oom             | 9690.61 \| 0.10 | 8344.36 \| 0.12 | oom                 |
| 1920       | oom             | oom             | oom             | oom                 |

Latency (sec)
| batch_size | HF (fp32) | HF (bf16) | HF (int8) | DS-inference (fp16) |
|:----------:|:---------:|:---------:|:---------:|:-------------------:|
| 1          | 1.94      | 2.11      | 6.05      | 1.62                |
| 2          | 1.92      | 2.06      | 5.92      | 1.65                |
| 4          | 1.89      | 2.06      | 5.94      | 1.67                |
| 8          | 1.94      | 2.16      | 5.96      | 1.62                |
| 16         | 1.99      | 2.05      | 5.80      | 1.65                |
| 32         | 2.03      | 2.08      | 5.96      | 1.60                |
| 64         | 2.36      | 2.11      | 5.98      | 1.61                |
| 128        | 4.30      | 2.21      | 6.23      | 1.70                |
| 256        | 6.93      | 3.12      | 7.26      | 2.50                |
| 384        | 10.69     | 4.12      | 8.37      | 3.46                |
| 512        | 14.82     | 5.42      | 9.45      | 4.49                |
| 640        | 19.85     | 6.69      | 10.47     | 5.51                |
| 768        | 20.18     | 8.11      | 11.67     | 6.50                |
| 896        | 24.53     | 9.45      | 12.60     | 7.63                |
| 1024       | oom       | 10.55     | 13.68     | 8.88                |
| 1152       | oom       | 11.86     | 15.27     | oom                 |
| 1280       | oom       | 13.24     | 16.29     | oom                 |
| 1408       | oom       | 14.41     | 17.35     | oom                 |
| 1536       | oom       | 15.76     | 18.73     | oom                 |
| 1664       | oom       | 17.12     | 20.22     | oom                 |
| 1792       | oom       | 18.49     | 21.48     | oom                 |
| 1920       | oom       | oom       | oom       | oom                 |

GPT2 Multi-Head Attention
```python
hidden_size = 2048
n_head = 16
n_layer = 24
total_params = 1315725312
```

Throughput (tokens/sec | msec/token)
| batch_size |    HF (fp32)    |    HF (bf16)     |    HF (int8)     | DS-inference (fp16) |
|:----------:|:---------------:|:----------------:|:----------------:|:-------------------:|
| 1          | 43.11 \| 23.20  | 40.69 \| 24.57   | 32.29 \| 30.97   | 122.76 \| 8.15      |
| 2          | 80.76 \| 12.38  | 80.87 \| 12.37   | 63.54 \| 15.74   | 247.85 \| 4.03      |
| 4          | 160.38 \| 6.24  | 154.98 \| 6.45   | 131.00 \| 7.63   | 503.52 \| 1.99      |
| 8          | 328.62 \| 3.04  | 332.90 \| 3.00   | 260.16 \| 3.84   | 1022.20 \| 0.98     |
| 16         | 662.08 \| 1.51  | 669.27 \| 1.49   | 523.29 \| 1.91   | 2027.35 \| 0.49     |
| 32         | 1314.92 \| 0.76 | 1287.95 \| 0.78  | 1055.57 \| 0.95  | 4231.82 \| 0.24     |
| 64         | 2118.17 \| 0.47 | 2487.35 \| 0.40  | 1969.26 \| 0.51  | 8311.39 \| 0.12     |
| 128        | 2860.26 \| 0.35 | 4268.99 \| 0.23  | 3581.49 \| 0.28  | 15879.15 \| 0.06    |
| 256        | 3487.86 \| 0.29 | 6917.01 \| 0.14  | 6132.47 \| 0.16  | 21635.49 \| 0.05    |
| 384        | 3794.16 \| 0.26 | 8821.31 \| 0.11  | 7774.37 \| 0.13  | 23872.25 \| 0.04    |
| 512        | 3804.37 \| 0.26 | 10068.51 \| 0.10 | 8872.88 \| 0.11  | 25009.06 \| 0.04    |
| 640        | 4124.01 \| 0.24 | 10547.88 \| 0.09 | 9956.58 \| 0.10  | oom                 |
| 768        | 3950.39 \| 0.25 | 10675.09 \| 0.09 | 10584.21 \| 0.09 | oom                 |
| 896        | 3937.28 \| 0.25 | 10780.82 \| 0.09 | 10994.00 \| 0.09 | oom                 |
| 1024       | oom             | 11192.55 \| 0.09 | 11306.37 \| 0.09 | oom                 |
| 1152       | oom             | 11178.30 \| 0.09 | 11290.51 \| 0.09 | oom                 |
| 1280       | oom             | 11383.98 \| 0.09 | 11459.89 \| 0.09 | oom                 |
| 1408       | oom             | 11477.66 \| 0.09 | 11565.90 \| 0.09 | oom                 |
| 1536       | oom             | 11382.66 \| 0.09 | 11491.99 \| 0.09 | oom                 |
| 1664       | oom             | 11571.52 \| 0.09 | 11603.73 \| 0.09 | oom                 |
| 1792       | oom             | 11394.20 \| 0.09 | 11412.46 \| 0.09 | oom                 |
| 1920       | oom             | oom              | oom              | oom                 |

Latency (sec)
| batch_size | HF (fp32) | HF (bf16) | HF (int8) | DS-inference (fp16) |
|:----------:|:---------:|:---------:|:---------:|:-------------------:|
| 1          | 2.32      | 2.46      | 3.10      | 0.81                |
| 2          | 2.48      | 2.47      | 3.15      | 0.81                |
| 4          | 2.49      | 2.58      | 3.05      | 0.79                |
| 8          | 2.43      | 2.40      | 3.07      | 0.78                |
| 16         | 2.42      | 2.39      | 3.06      | 0.79                |
| 32         | 2.43      | 2.48      | 3.03      | 0.76                |
| 64         | 3.02      | 2.57      | 3.25      | 0.77                |
| 128        | 4.48      | 3.00      | 3.57      | 0.81                |
| 256        | 7.34      | 3.70      | 4.17      | 1.18                |
| 384        | 10.12     | 4.35      | 4.94      | 1.61                |
| 512        | 13.46     | 5.09      | 5.77      | 2.05                |
| 640        | 15.52     | 6.07      | 6.43      | oom                 |
| 768        | 19.44     | 7.19      | 7.26      | oom                 |
| 896        | 22.76     | 8.31      | 8.15      | oom                 |
| 1024       | oom       | 9.15      | 9.06      | oom                 |
| 1152       | oom       | 10.31     | 10.20     | oom                 |
| 1280       | oom       | 11.24     | 11.17     | oom                 |
| 1408       | oom       | 12.27     | 12.17     | oom                 |
| 1536       | oom       | 13.49     | 13.37     | oom                 |
| 1664       | oom       | 14.38     | 14.34     | oom                 |
| 1792       | oom       | 15.73     | 15.70     | oom                 |
| 1920       | oom       | oom       | oom       | oom                 |
