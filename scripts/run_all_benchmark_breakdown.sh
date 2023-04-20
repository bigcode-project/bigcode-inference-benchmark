
# Santacoder
./scripts/run_benchmark_breakdown.sh santacoder bigcode/gpt_bigcode-santacoder 1 2040 5 0
./scripts/run_benchmark_breakdown.sh santacoder bigcode/gpt_bigcode-santacoder 32 2040 5 0
./scripts/run_benchmark_breakdown.sh santacoder bigcode/gpt_bigcode-santacoder 256 2040 5 0

./scripts/run_benchmark_breakdown.sh santacoder bigcode/gpt_bigcode-santacoder 1 2040 11 1
./scripts/run_benchmark_breakdown.sh santacoder bigcode/gpt_bigcode-santacoder 32 2040 11 1
./scripts/run_benchmark_breakdown.sh santacoder bigcode/gpt_bigcode-santacoder 256 2040 11 1

# Large model
./scripts/run_benchmark_breakdown.sh large_model ./data/large-model 1 8190 11 0
./scripts/run_benchmark_breakdown.sh large_model ./data/large-model 8 8190 11 0
./scripts/run_benchmark_breakdown.sh large_model ./data/large-model 32 8190 11 0
./scripts/run_benchmark_breakdown.sh large_model ./data/large-model 256 8190 11 0 # OOM?

./scripts/run_benchmark_breakdown.sh large_model ./data/large-model 1 8190 29 1
./scripts/run_benchmark_breakdown.sh large_model ./data/large-model 8 8190 29 1
./scripts/run_benchmark_breakdown.sh large_model ./data/large-model 32 8190 29 1
./scripts/run_benchmark_breakdown.sh large_model ./data/large-model 256 8190 29 1 # OOM?
