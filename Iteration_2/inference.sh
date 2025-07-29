python scripts/vllm_infer.py --model_name_or_path output/qwen2_5vl_lora_sft --template qwen2_vl --dataset lego_dataset_25_grnd --save_name lego_dataset_25_grnd.jsonl
python scripts/vllm_infer.py --model_name_or_path output/qwen2_5vl_lora_sft --template qwen2_vl --dataset lego_dataset_25_obj --save_name lego_dataset_25_obj.jsonl
python scripts/vllm_infer.py --model_name_or_path output/qwen2_5vl_lora_sft --template qwen2_vl --dataset lego_dataset_25_state --save_name lego_dataset_25_state.jsonl
