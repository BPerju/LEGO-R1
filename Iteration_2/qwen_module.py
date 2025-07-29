
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2VLForConditionalGeneration, AutoProcessor
from typing import Dict, Any, Union
from trl.data_utils import maybe_apply_chat_template
import torch

from vlm_modules.vlm_module import VLMBaseModule

class Qwen2VLModule(VLMBaseModule):
    def __init__(self):
        super().__init__()
        self.additional_output_cache = {}

    def get_vlm_key(self):
        return "qwen"

    def get_model_class(self, model_id: str, model_init_kwargs: dict):
        if "Qwen2-VL" in model_id:
            model_cls = Qwen2VLForConditionalGeneration
        elif "Qwen2.5-VL" in model_id:
            model_cls = Qwen2_5_VLForConditionalGeneration
        else:
            raise ValueError(f"Unsupported model: {model_id}")
        return model_cls
    
    def post_model_init(self, model, processing_class):
        pass
    
    def get_processing_class(self):
        return AutoProcessor
    
    def get_vision_modules_keywords(self):  
        return ['visual']
    
    def get_custom_multimodal_keywords(self):
        return ['pixel_values', 'image_grid_thw']

    def get_non_generate_params(self):
        return []
    
    def get_custom_processing_keywords(self):
        return [('image_processor', 'max_pixels'), ('image_processor', 'min_pixels')]
    
    def prepare_prompt(self, processing_class, inputs: dict[str, Union[torch.Tensor, Any]]):
        prompts_text = [maybe_apply_chat_template(example, processing_class)["prompt"] for example in inputs]
        return prompts_text
        
    def prepare_model_inputs(
            self,
            processing_class,
            prompts_text, # List[str]: Prompts for the batch, one per sample
            images,       # List[PIL.Image] or None: Flattened list of images for the entire batch, or None
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False
    ):
        """
        Prepares inputs for the model and generates additional_output metadata.
    
        Crucially, the returned `additional_output` list must have the same length
        as `prompts_text` to satisfy the trainer's assertion.
        """
        # Reset cache every call
        self.additional_output_cache.clear()
    
        if images and len(images) > 0:
            prompt_inputs = processing_class(
                text=prompts_text,
                images=images, 
                return_tensors=return_tensors,
                padding=padding,
                padding_side=padding_side,
                add_special_tokens=add_special_tokens,
            )

            self.additional_output_cache["image_grid_thw"] = prompt_inputs.get("image_grid_thw", [])
            self.additional_output_cache["image_paths"] = images # Cache the PIL Images
    
            num_samples = len(prompts_text)
    
            batch_image_grid_thw = prompt_inputs.get("image_grid_thw", [])
            additional_output = [
                {
                    "image_grid_thw": batch_image_grid_thw, 
                    "image_paths": images                    
                }
                for _ in range(num_samples) 
            ]
    
        else: 
            prompt_inputs = processing_class(
                text=prompts_text,
                return_tensors=return_tensors,
                padding=padding,
                padding_side=padding_side,
                add_special_tokens=add_special_tokens,
            )
      
            num_samples = len(prompts_text)
            additional_output = [
                {
                    "image_grid_thw": [],
                    "image_paths": []
                }
                for _ in range(num_samples)
            ]

        return prompt_inputs, additional_output


    
    @staticmethod
    def get_question_template(task_type: str):
        match task_type:
            case "rec":
                return "{Question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags."
            case "ic":
                return "{Question} First thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> json format answer here </answer>"
            case "odLength":
                SYSTEM_PROMPT = (
                    "First thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
                    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
                    "<think> reasoning process here </think><answer> answer here </answer>"
                )
                return SYSTEM_PROMPT + '\n' + "{Question}"
            case _:
                return "{Question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags."


    @staticmethod
    def _infer_task_type_from_solution(solution_text: str) -> str:
        import re
        """Infers the task type based on the format of the ground truth solution text.
    
        Args:
            solution_text (str): The ground truth solution string for a sample.
    
        Returns:
            str: The inferred task type ('object', 'state', 'grounding', or 'unknown').
        """
        if not solution_text or not isinstance(solution_text, str):
            return "unknown"
    
        stripped_sol = solution_text.strip()
    
        if re.fullmatch(r'<\s*answer\s*>\s*\[\s*{.*?}\s*\]\s*<\s*/\s*answer\s*>', stripped_sol, flags=re.DOTALL | re.IGNORECASE):
            return "object"
  
        answer_content_match = re.search(r'<\s*answer\s*>(.*?)<\s*/\s*answer\s*>', stripped_sol, flags=re.DOTALL | re.IGNORECASE)
        if answer_content_match:
            answer_content = answer_content_match.group(1).strip()
            if re.fullmatch(r'(yes|no)\.?', answer_content, flags=re.IGNORECASE):
                 return "state"
    
        return "grounding"
            
    @staticmethod
    def format_reward(completions, solution, **kwargs):
        """
        Calculates a unified format reward based on task type inferred from the solution.
        - Object Task: Checks <think> (0.25), <answer> (0.25), JSON validity (0.25), 4 coordinates per box (0.25).
        - State Task: Checks only mandatory <think> (0.5) and <answer> (0.5) tags.
        - Grounding Task: Checks only mandatory <think> (0.5) and <answer> (0.5) tags.
        """
        import os, re, json
        from datetime import datetime
        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        dbg = os.getenv("DEBUG_MODE") == "true"
        logf = os.getenv("LOG_PATH")
    
        completion_contents = [completion[0]["content"] for completion in completions]
    
        if not solution:
            inferred_task_type_log = "unknown"
            first_solution_sample_log = "N/A"
        else:
            first_solution_sample_log = solution[0] if isinstance(solution[0], str) else str(solution[0])
            inferred_task_type_log = Qwen2VLModule._infer_task_type_from_solution(first_solution_sample_log)
    
        final_rewards = []
        detailed_logs = [] 
    
        for content in completion_contents:
            reward = 0.0
            has_think = 0.0
            has_answer = 0.0
            json_valid = 0.0 
            coords_valid = 0.0 
            parsed_data = None
            log_details = {"content_snippet": repr(content[:200]), "checks": {}}
    
            if inferred_task_type_log in ["state", "grounding"]:
                think_weight = 0.5
                answer_weight = 0.5
                json_weight = 0.0
                coords_weight = 0.0
            else: 
                think_weight = 0.25
                answer_weight = 0.25
                json_weight = 0.25
                coords_weight = 0.25
    
            if re.search(r"<think>.*?</think>", content, re.DOTALL):
                has_think = think_weight
                reward += has_think
            log_details["checks"]["has_think"] = has_think
    
            answer_match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)
            if answer_match:
                has_answer = answer_weight
                reward += has_answer
                answer_text = answer_match.group(1).strip()
                log_details["checks"]["has_answer"] = has_answer
    
                if inferred_task_type_log == "object":

                    try:
                        parsed_data = json.loads(answer_text)
                        if isinstance(parsed_data, list):
                            json_valid = json_weight
                            reward += json_valid
                        log_details["checks"]["json_valid"] = json_valid
                    except json.JSONDecodeError:
                        pass
    
                    if parsed_data is not None and isinstance(parsed_data, list):
                        all_coords_valid = True
                        for item in parsed_data:
                            if not (isinstance(item, dict) and 'bbox_2d' in item):
                                all_coords_valid = False
                                break
                            bbox = item['bbox_2d']
                            if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4 and all(isinstance(coord, (int, float)) for coord in bbox)):
                                all_coords_valid = False
                                break
                        if all_coords_valid:
                            coords_valid = coords_weight
                            reward += coords_valid
                        log_details["checks"]["coords_valid"] = coords_valid

            final_reward = min(reward, 1.0)
            final_rewards.append(final_reward)
            log_details["final_reward"] = final_reward
            detailed_logs.append(log_details)
    
        if dbg and logf:
            base_log_path = logf.replace(".txt", "_format_unified.txt") if logf else "debug_format_unified.txt"
            try:
                with open(base_log_path, "a", encoding='utf-8') as f:
                    f.write(f"\n" + "=" * 20 + f" {current_time} Format Reward (Task: {inferred_task_type_log}) " + "=" * 20 + "\n")
                    f.write(f"[Batch Info] Inferred Task Type: {inferred_task_type_log}\n")
                    f.write(f"[Batch Info] Based on Solution Sample (first): {repr(first_solution_sample_log[:100])}...\n")
                    f.write(f"[Batch Info] Number of Samples in Batch: {len(completion_contents)}\n")
    
                    for i, log_detail in enumerate(detailed_logs):
                        f.write(f"- Sample {i} Details -\n")
                        f.write(f"[Sample {i}] Raw Model Output Snippet: {log_detail['content_snippet']}...\n")
                        sol_snippet_for_sample = "N/A"
                        if solution and i < len(solution):
                            sol_text = solution[i] if isinstance(solution[i], str) else str(solution[i])
                            sol_snippet_for_sample = repr(sol_text[:100]) + "..."
                        f.write(f"[Sample {i}] Ground Truth Snippet: {sol_snippet_for_sample}\n")
                        for check_name, check_reward in log_detail['checks'].items():
                            f.write(f"[Sample {i}] {check_name}: {check_reward}\n")
                        f.write(f"[Sample {i}] Final Reward: {log_detail['final_reward']}\n")
                        f.write(f"- End Sample {i} Details -\n")
                    f.write("- End Final Results Summary -\n")
                    f.write("=" * 20 + f" End Format Reward (Task: {inferred_task_type_log}) Log " + "=" * 20 + "\n")
            except Exception as log_error:
                print(f"[Format Reward Logging Error]: {log_error}")
    
        return final_rewards
        
    @staticmethod
    def iou_reward(completions, solution, **kwargs):
        """
        Calculates a combined reward based on IoU and ROUGE-1 for labels.
        Applies to samples inferred to be of 'object' task type based on solution format.
        Final reward = min(1.0, (iou_reward + rouge1_reward) / 2.0)
        Expected format for Ground Truth (solution[i]):
            String wrapped in <answer> tags containing JSON list.
            "<answer>[{'bbox_2d': [x1, y1, x2, y2], 'label': 'object name'}, ...]</answer>"
        Expected format for Model Prediction (completions[i][0]['content']):
            Text containing "<answer>[{'bbox_2d': [x1, y1, x2, y2], 'label': '...'}, ...]</answer>"
        """
        import re, os, json, numpy as np
        from scipy.optimize import linear_sum_assignment
        try:
            from rouge import Rouge
            rouge_available = True
            rouge_scorer = Rouge()
        except ImportError:
            rouge_available = False
            rouge_scorer = None
        completion_contents = [completion[0]["content"] for completion in completions]

        def iou(a, b):
            """Calculates IoU for two boxes [x1, y1, x2, y2]."""
            ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
            ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            ua = (a[2] - a[0]) * (a[3] - a[1])
            ub = (b[2] - b[0]) * (b[3] - b[1])
            return inter / (ua + ub - inter) if ua + ub - inter else 0.0

        def parse_boxes_and_labels(raw: str):
            """Parses JSON list-of-dicts format, returning boxes and labels."""
            raw = raw.strip() 
            try:
                data = json.loads(raw)
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON parsing failed: {e}")
            if not isinstance(data, list):
                raise ValueError("Parsed data is not a list")
            boxes = []
            labels = []
            for item in data: # Use 'data' from json.loads
                if not (isinstance(item, dict) and 'bbox_2d' in item and 'label' in item):
                    raise ValueError("Item missing 'bbox_2d' or 'label'")
                bbox = item['bbox_2d']
                if not (isinstance(bbox, list) and len(bbox) == 4):
                    raise ValueError("'bbox_2d' is not a list of 4 elements")
                try:
                    x1, y1, x2, y2 = map(float, bbox)
                except ValueError:
                    raise ValueError("'bbox_2d' coordinates invalid")
                if x2 <= x1 or y2 <= y1:
                    raise ValueError("degenerate box")
                boxes.append([x1, y1, x2, y2])
                labels.append(str(item['label']).strip())
            return boxes, labels

        pred_raw = [c[0]["content"] for c in completions]
        rewards = []
        dbg = str(os.getenv("DEBUG_MODE", "")).lower() in {"true", "1", "yes"}
        logf = os.getenv("LOG_PATH")

        def _d(msg: str):
            """Debug logging helper."""
            if dbg and logf:
                try:
                    with open(logf, "a", encoding="utf-8") as fh: 
                        fh.write(msg + "\n")
                except Exception as log_error:
                    print(f"[IoU/ROUGE Debug Logging Error]: {log_error}")

        if not solution:
            inferred_task_type_log = "unknown"
            first_solution_sample_log = "N/A"
        else:
            first_solution_sample_log = solution[0] if isinstance(solution[0], str) else str(solution[0])
            inferred_task_type_log = Qwen2VLModule._infer_task_type_from_solution(first_solution_sample_log)
        final_rewards = []
        if inferred_task_type_log != "object":
            final_rewards = [0.0] * len(completion_contents)
            return final_rewards
        for i, (p_txt, s_txt) in enumerate(zip(pred_raw, solution)):
            reward = 0.0
            mean_iou = 0.0
            mean_rouge1 = 0.0
            bonus = 0.0
            gt_boxes_final = []
            pred_boxes_final = []
            gt_labels_final = []
            pred_labels_final = []
            matched_gt_indices = []
            matched_pred_indices = []
            parsing_error = None
            try:
                gt_match = re.search(r'<answer>(.*?)</answer>', s_txt, re.DOTALL)
                if not gt_match:
                    raise ValueError("no <answer> tag found in GT")
                gt_boxes_final, gt_labels_final = parse_boxes_and_labels(gt_match.group(1).strip())

                pred_match = re.search(r'<answer>(.*?)</answer>', p_txt, re.DOTALL)
                if not pred_match:
                    raise ValueError("no <answer> tag found in prediction")
                pred_boxes_final, pred_labels_final = parse_boxes_and_labels(pred_match.group(1).strip())

                g, p = len(gt_boxes_final), len(pred_boxes_final)

                if g == 0 and p == 0:
                    mean_iou = 1.0
                    mean_rouge1 = 1.0 if rouge_available else 0.0
                elif g == 0 or p == 0:
                    mean_iou = 0.0
                    mean_rouge1 = 0.0
                else:
                    cost = np.zeros((g, p))
                    for r, gt in enumerate(gt_boxes_final):
                        for c, pr in enumerate(pred_boxes_final):
                            try:
                                cost[r, c] = 1.0 - iou(gt, pr)
                            except Exception as iou_error:
                                _d(f"[Sample {i}] [IoU Calc Error] GT {gt} Pred {pr}: {iou_error}")
                                cost[r, c] = 1.0

                    try:
                        rows, cols = linear_sum_assignment(cost)
                    except Exception as hungarian_error:
                        _d(f"[Sample {i}] [Hungarian Error]: {hungarian_error}")
                        rows, cols = np.array([]), np.array([])

                    if rows.size == 0 or cols.size == 0:
                        mean_iou = 0.0
                        mean_rouge1 = 0.0
                    else:
                        matched_ious = 1.0 - cost[rows, cols]
                        mean_iou = float(np.clip(matched_ious.mean(), 0.0, 1.0))
                        matched_gt_indices = rows.tolist()
                        matched_pred_indices = cols.tolist()

                        if rouge_available and len(matched_gt_indices) > 0:
                            matched_gt_labels = [gt_labels_final[idx] for idx in matched_gt_indices]
                            matched_pred_labels = [pred_labels_final[idx] for idx in matched_pred_indices]

                            rouge_scores = []
                            for gt_lbl, pred_lbl in zip(matched_gt_labels, matched_pred_labels):
                                if gt_lbl and pred_lbl:
                                    try:
                                        scores = rouge_scorer.get_scores(hyps=[pred_lbl], refs=[gt_lbl])
                                        rouge1_f1 = scores[0]['rouge-1']['f']
                                        rouge_scores.append(rouge1_f1)
                                    except Exception as rouge_error:
                                        _d(f"[Sample {i}] [ROUGE Calc Error] GT: '{gt_lbl}' Pred: '{pred_lbl}' Error: {rouge_error}")
                                        rouge_scores.append(0.0)
                                elif not gt_lbl and not pred_lbl:
                                    rouge_scores.append(1.0)
                                else:
                                    rouge_scores.append(0.0)

                            if rouge_scores:
                                mean_rouge1 = float(np.clip(np.mean(rouge_scores), 0.0, 1.0))
                            else:
                                mean_rouge1 = 0.0

                mean_iou = float(np.clip(mean_iou, 0.0, 1.0)) if np.isfinite(mean_iou) else 0.0
                mean_rouge1 = float(np.clip(mean_rouge1, 0.0, 1.0)) if np.isfinite(mean_rouge1) else 0.0
                bonus = 0.1 if g == p and g > 0 else 0.0
                combined_core_reward = (mean_iou + mean_rouge1) / 2.0
                reward_before_cap = combined_core_reward + bonus
                reward = min(reward_before_cap, 1.0)

            except Exception as e:
                parsing_error = e
                _d(f"[Sample {i}] [Error] Processing failed: {e}")
                reward = 0.0 

            if not np.isfinite(reward):
                _d(f"[Sample {i}] [Warning] final reward {reward} -> 0")
                reward = 0.0
            rewards.append(reward)

            _d("=" * 20 + f" [Sample {i}] IoU/ROUGE Debug Info (Task: {inferred_task_type_log}) " + "=" * 20)
            if parsing_error:
                _d(f"[Sample {i}] Parsing/Calculation Error: {parsing_error}")
            _d(f"[Sample {i}] Ground Truth Snippet : {repr(s_txt[:100])}...")
            _d(f"[Sample {i}] Model Pred Snippet : {repr(p_txt[:100])}...")
            _d(f"[Sample {i}] Parsed GT Boxes : {gt_boxes_final}")
            _d(f"[Sample {i}] Parsed Pred Boxes : {pred_boxes_final}")
            _d(f"[Sample {i}] Parsed GT Labels : {gt_labels_final}")
            _d(f"[Sample {i}] Parsed Pred Labels : {pred_labels_final}")
            _d(f"[Sample {i}] Matched GT Indices : {matched_gt_indices}")
            _d(f"[Sample {i}] Matched Pred Indices: {matched_pred_indices}")
            _d(f"[Sample {i}] Mean IoU : {mean_iou:.4f}")
            _d(f"[Sample {i}] Mean ROUGE-1 (F1) : {mean_rouge1:.4f}")
            _d(f"[Sample {i}] Count Bonus : {bonus:.2f}")
            _d(f"[Sample {i}] Combined Core Reward: {(mean_iou + mean_rouge1) / 2.0:.4f}")
            _d(f"[Sample {i}] Final Reward : {reward:.4f}")
            _d("=" * 20 + f" [Sample {i}] End Debug Info (Task: {inferred_task_type_log}) " + "=" * 20 + "\n")

        return rewards

    @staticmethod
    def state_reward(completions, solution, **kwargs):
        """
        Calculates accuracy reward for 'state' task samples.
        Reward = 1.0 if prediction ('Yes'/'No') matches ground truth.
        Reward = 0.0 otherwise or if task is not 'state'.
        Applies to samples inferred to be of 'state' task type based on solution format.
        """
        import os, re
        from datetime import datetime
    
        completion_contents = [completion[0]["content"] for completion in completions]
        rewards = []
        dbg = str(os.getenv("DEBUG_MODE", "")).lower() in {"true", "1", "yes"}
        logf = os.getenv("LOG_PATH")
    
        def _d(msg: str):
            """Debug logging helper."""
            if dbg and logf:
                try:
                    with open(logf, "a", encoding="utf-8") as fh:
                        fh.write(msg + "\n")
                except Exception as log_error:
                    print(f"[State Reward Debug Logging Error]: {log_error}")
    
        if not solution:
            inferred_task_type_log = "unknown"
            first_solution_sample_log = "N/A"
        else:
            first_solution_sample_log = solution[0] if isinstance(solution[0], str) else str(solution[0])
            inferred_task_type_log = Qwen2VLModule._infer_task_type_from_solution(first_solution_sample_log)
    
        if inferred_task_type_log != "state":
            final_rewards = [0.0] * len(completion_contents)
            if dbg and logf:
                 skip_log_path = logf.replace(".txt", "_state_acc_skipped.txt") if logf else "debug_state_acc_skipped.txt"
                 try:
                     with open(skip_log_path, "a", encoding='utf-8') as f:
                         f.write(f"\n--- {datetime.now().strftime('%d-%H-%M-%S-%f')} State Acc Reward Skipped (Task: {inferred_task_type_log}) ---\n")
                         f.write(f"[Batch] Skipped because task type '{inferred_task_type_log}' != 'state'.\n")
                         f.write(f"[Batch] Based on Solution Sample (first): {repr(first_solution_sample_log[:100])}...\n")
                         f.write(f"[Batch] Number of samples skipped: {len(completion_contents)}\n")
                         f.write("--- End Skipped Batch Log ---\n")
                 except Exception as log_error:
                     print(f"[State Acc Reward Skipped Logging Error]: {log_error}")
            return final_rewards

        def normalize_answer(text):
            """Normalize Yes/No answer (strip, lowercase, remove period)."""
            if not isinstance(text, str):
                return ""
            return re.sub(r'\.?$', '', text.strip().lower())
    
        for i, (p_txt, s_txt) in enumerate(zip(completion_contents, solution)):
            reward = 0.0
            parsing_error = None
            try:

                gt_match = re.search(r"<\s*answer\s*>(.*?)<\s*/\s*answer\s*>", s_txt, re.DOTALL)
                gt_text_to_check = gt_match.group(1).strip() if gt_match else s_txt.strip()
                norm_gt = normalize_answer(gt_text_to_check)

                pred_match = re.search(r"<\s*answer\s*>(.*?)<\s*/\s*answer\s*>", p_txt, re.DOTALL)
                pred_text_to_check = pred_match.group(1).strip() if pred_match else p_txt.strip()
                norm_pred = normalize_answer(pred_text_to_check)

                if norm_pred in ('yes', 'no'):
                    reward = 1.0 if norm_pred == norm_gt else 0.0 
                else:
                    reward = 0.0
    
            except Exception as e:
                parsing_error = e
                _d(f"[Sample {i}] [Error] State Acc Processing failed: {e}")
                reward = 0.0 
            rewards.append(reward)
    
            _d("=" * 20 + f" [Sample {i}] State Acc Reward Debug Info (Task: {inferred_task_type_log}) " + "=" * 20)
            if parsing_error:
                _d(f"[Sample {i}] Parsing/Calculation Error: {parsing_error}")
            _d(f"[Sample {i}] Ground Truth Snippet : {repr(s_txt[:100])}...")
            _d(f"[Sample {i}] Model Pred Snippet : {repr(p_txt[:100])}...")
            _d(f"[Sample {i}] Normalized GT : '{norm_gt}'")
            _d(f"[Sample {i}] Pred Text to Check : '{pred_text_to_check[:100]}...'") 
            _d(f"[Sample {i}] Normalized Pred : '{norm_pred}'")
            _d(f"[Sample {i}] Match : {norm_pred == norm_gt}")
            _d(f"[Sample {i}] Final Reward : {reward:.4f}")
            _d("=" * 20 + f" [Sample {i}] End State Acc Debug Info (Task: {inferred_task_type_log}) " + "=" * 20 + "\n")
    
        return rewards
    
    @staticmethod
    def grounding_reward(completions, solution, **kwargs):
        """
        Calculates accuracy reward for 'grounding' task samples using ROUGE-1 F1 and BLEU-4 scores.
        Reward = Sum of (ROUGE-1 F1, BLEU-4) scores (capped at 2.0) between predicted and ground truth text.
        Applies to samples inferred to be of 'grounding' task type based on solution format.
        """
        import os, re
        from datetime import datetime
        
        try:
            from rouge import Rouge
            rouge_available = True
            rouge_scorer = Rouge()
        except ImportError:
            rouge_available = False
            rouge_scorer = None
    
        try:
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            import nltk
            nltk.download('punkt', quiet=True) 
            nltk.download('punkt_tab',quiet=True)
            bleu_available = True
            smoothing = SmoothingFunction().method1 
        except ImportError:
            bleu_available = False
            smoothing = None
    
        completion_contents = [completion[0]["content"] for completion in completions]
        rewards = []
        dbg = str(os.getenv("DEBUG_MODE", "")).lower() in {"true", "1", "yes"}
        logf = os.getenv("LOG_PATH")
    
        def _d(msg: str):
            """Debug logging helper."""
            if dbg and logf:
                try:
                    with open(logf, "a", encoding="utf-8") as fh:
                        fh.write(msg + "\n")
                except Exception as log_error:
                    print(f"[Grounding Reward Debug Logging Error]: {log_error}")
    
        if not solution:
            inferred_task_type_log = "unknown"
            first_solution_sample_log = "N/A"
        else:
            first_solution_sample_log = solution[0] if isinstance(solution[0], str) else str(solution[0])
            inferred_task_type_log = Qwen2VLModule._infer_task_type_from_solution(first_solution_sample_log)
    

        if inferred_task_type_log != "grounding":
            final_rewards = [0.0] * len(completion_contents)
            if dbg and logf:
                 skip_log_path = logf.replace(".txt", "_grounding_acc_skipped.txt") if logf else "debug_grounding_acc_skipped.txt"
                 try:
                     with open(skip_log_path, "a", encoding='utf-8') as f:
                         f.write(f"\n--- {datetime.now().strftime('%d-%H-%M-%S-%f')} Grounding Acc Reward Skipped (Task: {inferred_task_type_log}) ---\n")
                         f.write(f"[Batch] Skipped because task type '{inferred_task_type_log}' != 'grounding'.\n")
                         f.write(f"[Batch] Based on Solution Sample (first): {repr(first_solution_sample_log[:100])}...\n")
                         f.write(f"[Batch] Number of samples skipped: {len(completion_contents)}\n")
                         f.write("--- End Skipped Batch Log ---\n")
                 except Exception as log_error:
                     print(f"[Grounding Acc Reward Skipped Logging Error]: {log_error}")
            return final_rewards

        for i, (p_txt, s_txt) in enumerate(zip(completion_contents, solution)):
            reward = 0.0
            rouge_reward = 0.0
            bleu_reward = 0.0
            parsing_error = None
            try:

                pred_match = re.search(r"<\s*answer\s*>(.*?)<\s*/\s*answer\s*>", p_txt, re.DOTALL)
                pred_text = pred_match.group(1).strip() if pred_match else p_txt.strip()
    
                gt_match = re.search(r"<\s*answer\s*>(.*?)<\s*/\s*answer\s*>", s_txt, re.DOTALL)
                gt_text = gt_match.group(1).strip() if gt_match else s_txt.strip()
    
                if pred_text and gt_text:
                    if rouge_available:
                        try:
                            scores = rouge_scorer.get_scores(hyps=[pred_text], refs=[gt_text])
                            rouge1_f1 = scores[0]['rouge-1']['f']
                            rouge_reward = float(rouge1_f1) if 0.0 <= rouge1_f1 <= 1.0 else 0.0
                        except Exception as rouge_error:
                            parsing_error = rouge_error 
                            _d(f"[Sample {i}] [ROUGE Calc Error] Pred: '{pred_text[:50]}...' GT: '{gt_text[:50]}...' Error: {rouge_error}")
                            rouge_reward = 0.0
                    
                    if bleu_available:
                        try:

                            from nltk import word_tokenize
                            ref_tokens = word_tokenize(gt_text.lower())
                            hyp_tokens = word_tokenize(pred_text.lower())
                            
                            weights = (0.25, 0.25, 0.25, 0.25)
                            
                            bleu_score = sentence_bleu(
                                [ref_tokens],
                                hyp_tokens,  
                                weights=weights,
                                smoothing_function=smoothing
                            )
                            bleu_reward = float(bleu_score) if 0.0 <= bleu_score <= 1.0 else 0.0
                            
                        except Exception as bleu_error:
                            parsing_error = bleu_error 
                            _d(f"[Sample {i}] [BLEU Calc Error] Pred: '{pred_text[:50]}...' GT: '{gt_text[:50]}...' Error: {bleu_error}")
                            bleu_reward = 0.0
    
                    reward = rouge_reward + bleu_reward
    
    
            except Exception as e:
                parsing_error = e
                _d(f"[Sample {i}] [Error] Grounding Acc Processing failed: {e}")
                reward = 0.0 
    
            if not isinstance(reward, (int, float)):
                 reward = 0.0
            else:
                 reward = min(reward, 2.0)
                 reward = max(reward, 0.0)
    
            rewards.append(reward)
    
            _d("=" * 20 + f" [Sample {i}] Grounding Acc Reward Debug Info (Task: {inferred_task_type_log}) " + "=" * 20)
            if parsing_error and "Calc Error" not in str(parsing_error): 
                _d(f"[Sample {i}] Parsing/Calculation Error: {parsing_error}")
            _d(f"[Sample {i}] Ground Truth Snippet : {repr(s_txt[:100])}...")
            _d(f"[Sample {i}] Model Pred Snippet : {repr(p_txt[:100])}...")
            _d(f"[Sample {i}] Extracted GT Text : '{gt_text[:100]}...'")
            _d(f"[Sample {i}] Extracted Pred Text : '{pred_text[:100]}...'")
            _d(f"[Sample {i}] ROUGE-1 F1 Reward : {rouge_reward:.4f}")
            _d(f"[Sample {i}] BLEU-4 Reward : {bleu_reward:.4f}")
            _d(f"[Sample {i}] Final Combined Reward (Sum, capped @ 2.0) : {reward:.4f}") 
            _d("=" * 20 + f" [Sample {i}] End Grounding Acc Debug Info (Task: {inferred_task_type_log}) " + "=" * 20 + "\n")
    
        return rewards
            
    @staticmethod
    def select_reward_func(func: str, task_type: str):
        """
        Selects the appropriate reward function.
        Note: task_type is from command line (--task_type), but functions now infer task dynamically.
        """
        if func == "accuracy":
            if task_type == "rec": 
                return Qwen2VLModule.iou_reward 
            else:
                raise ValueError(f"Unsupported task type '{task_type}' for accuracy reward func '{func}'.")
        elif func == "format":
            if task_type == "rec": 
                return Qwen2VLModule.format_reward 
            else:
                raise ValueError(f"Unsupported task type '{task_type}' for format reward func '{func}'.")

        elif func == "state_acc": 
             return Qwen2VLModule.state_reward
        elif func == "grnd": 
             return Qwen2VLModule.grounding_reward
        
        else:
            raise ValueError(f"Unsupported reward function '{func}' for task type '{task_type}'.")