# bigcode-inference-benchmark
A100 80GB

## BLOOM
```python
hidden_size = 2048
n_head = 16
n_layer = 24
total_params = 1311535104
```

Throughput (tokens/sec | msec/token)
| batch_size |    HF (fp32)    |    HF (bf16)     |    HF (int8)    | DS-inference (fp16) |
|:----------:|:---------------:|:----------------:|:---------------:|:-------------------:|
| 1          | 77.94 \| 12.83  | 72.50 \| 13.79   | 20.94 \| 47.75  | 104.00 \| 9.62      |
| 2          | 155.77 \| 6.42  | 143.44 \| 6.97   | 41.44 \| 24.13  | 206.33 \| 4.85      |
| 4          | 319.15 \| 3.13  | 293.06 \| 3.41   | 83.02 \| 12.04  | 418.28 \| 2.39      |
| 8          | 596.68 \| 1.68  | 581.10 \| 1.72   | 167.03 \| 5.99  | 828.67 \| 1.21      |
| 16         | 1146.25 \| 0.87 | 1147.91 \| 0.87  | 330.12 \| 3.03  | 1652.51 \| 0.61     |
| 32         | 2177.47 \| 0.46 | 2356.71 \| 0.42  | 673.33 \| 1.49  | 3280.17 \| 0.30     |
| 64         | 2776.93 \| 0.36 | 4784.46 \| 0.21  | 1329.42 \| 0.75 | 6717.77 \| 0.15     |
| 128        | 3007.26 \| 0.33 | 8056.59 \| 0.12  | 2491.86 \| 0.40 | 10410.82 \| 0.10    |
| 256        | 3758.11 \| 0.27 | 10339.00 \| 0.10 | 4325.98 \| 0.23 | 12707.62 \| 0.08    |
| 384        | 3658.51 \| 0.27 | 11091.67 \| 0.09 | 5628.15 \| 0.18 | 13483.54 \| 0.07    |
| 512        | 3775.92 \| 0.26 | 11332.58 \| 0.09 | 6675.52 \| 0.15 | 13930.89 \| 0.07    |
| 640        | 3938.85 \| 0.25 | 11534.74 \| 0.09 | 7472.39 \| 0.13 | 14399.86 \| 0.07    |
| 768        | 3886.59 \| 0.26 | 11354.37 \| 0.09 | 8220.54 \| 0.12 | 14656.84 \| 0.07    |
| 896        | 3728.33 \| 0.27 | 11286.69 \| 0.09 | 8686.16 \| 0.12 | 14540.19 \| 0.07    |
| 1024       | oom             | 11692.32 \| 0.09 | 9012.79 \| 0.11 | 14390.77 \| 0.07    |
| 1152       | oom             | 11894.50 \| 0.08 | 9147.50 \| 0.11 | oom                 |
| 1280       | oom             | 11731.85 \| 0.09 | 9507.04 \| 0.11 | oom                 |
| 1408       | oom             | 11802.63 \| 0.08 | 9711.69 \| 0.10 | oom                 |
| 1536       | oom             | 11857.12 \| 0.08 | 9873.34 \| 0.10 | oom                 |
| 1664       | oom             | 11932.68 \| 0.08 | 9756.13 \| 0.10 | oom                 |
| 1792       | oom             | 11653.63 \| 0.09 | 9814.68 \| 0.10 | oom                 |
| 1920       | oom             | oom              | oom             | oom                 |

Latency (sec)
| batch_size | HF (fp32) | HF (bf16) | HF (int8) | DS-inference (fp16) |
|:----------:|:---------:|:---------:|:---------:|:-------------------:|
| 1          | 1.28      | 1.38      | 4.77      | 0.96                |
| 2          | 1.28      | 1.39      | 4.83      | 0.97                |
| 4          | 1.25      | 1.36      | 4.82      | 0.96                |
| 8          | 1.34      | 1.38      | 4.79      | 0.97                |
| 16         | 1.40      | 1.39      | 4.85      | 0.97                |
| 32         | 1.47      | 1.36      | 4.75      | 0.98                |
| 64         | 2.30      | 1.34      | 4.81      | 0.95                |
| 128        | 4.26      | 1.59      | 5.14      | 1.23                |
| 256        | 6.81      | 2.48      | 5.92      | 2.01                |
| 384        | 10.50     | 3.46      | 6.82      | 2.85                |
| 512        | 13.56     | 4.52      | 7.67      | 3.68                |
| 640        | 16.25     | 5.55      | 8.56      | 4.44                |
| 768        | 19.76     | 6.76      | 9.34      | 5.24                |
| 896        | 24.03     | 7.94      | 10.32     | 6.16                |
| 1024       | oom       | 8.76      | 11.36     | 7.12                |
| 1152       | oom       | 9.69      | 12.59     | oom                 |
| 1280       | oom       | 10.91     | 13.46     | oom                 |
| 1408       | oom       | 11.93     | 14.50     | oom                 |
| 1536       | oom       | 12.95     | 15.56     | oom                 |
| 1664       | oom       | 13.94     | 17.06     | oom                 |
| 1792       | oom       | 15.38     | 18.26     | oom                 |
| 1920       | oom       | oom       | oom       | oom                 |

## GPT2 Multi-Head Attention
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
