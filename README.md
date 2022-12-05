# bigcode-inference-benchmark
A100 80GB

BLOOM\
```python
hidden_size = 2048
n_head = 16
n_layer = 24
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