
# Santacoder
./scripts/run_textgen_benchmark_breakdown.sh santacoder bigcode/gpt_bigcode-santacoder 1 2040 5 0 v2_
./scripts/run_textgen_benchmark_breakdown.sh santacoder bigcode/gpt_bigcode-santacoder 32 2040 5 0 v2_
./scripts/run_textgen_benchmark_breakdown.sh santacoder bigcode/gpt_bigcode-santacoder 256 2040 5 0 v2_

./scripts/run_textgen_benchmark_breakdown.sh santacoder bigcode/gpt_bigcode-santacoder 1 2040 11 1 v2_
./scripts/run_textgen_benchmark_breakdown.sh santacoder bigcode/gpt_bigcode-santacoder 32 2040 11 1 v2_
./scripts/run_textgen_benchmark_breakdown.sh santacoder bigcode/gpt_bigcode-santacoder 256 2040 11 1 v2_

# Large model
./scripts/run_textgen_benchmark_breakdown.sh large_model ./data/bigcode_large-model 1 8190 11 0 v2_
./scripts/run_textgen_benchmark_breakdown.sh large_model ./data/bigcode_large-model 8 8190 11 0 v2_
./scripts/run_textgen_benchmark_breakdown.sh large_model ./data/bigcode_large-model 32 8190 11 0 v2_
./scripts/run_textgen_benchmark_breakdown.sh large_model ./data/bigcode_large-model 256 8190 11 0  v2_ # OOM?

./scripts/run_textgen_benchmark_breakdown.sh large_model ./data/bigcode_large-model 1 8190 29 1 v2_ 1
./scripts/run_textgen_benchmark_breakdown.sh large_model ./data/bigcode_large-model 8 8190 29 1 v2_ 1
./scripts/run_textgen_benchmark_breakdown.sh large_model ./data/bigcode_large-model 32 8190 29 1 v2_ 1
./scripts/run_textgen_benchmark_breakdown.sh large_model ./data/bigcode_large-model 256 8190 29 1  v2_ 1 # OOM?
