
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
            prompts_text,
            images,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False):
    
        # reset cache every call
        self.additional_output_cache.clear()
    
        if images:
            prompt_inputs = processing_class(
                text=prompts_text,
                images=images,
                return_tensors=return_tensors,
                padding=padding,
                padding_side=padding_side,
                add_special_tokens=add_special_tokens,
            )
    
            # cache & per‑sample extras
            self.additional_output_cache["image_grid_thw"] = prompt_inputs["image_grid_thw"]
            self.additional_output_cache["image_paths"] = images
            additional_output = [
                {"image_grid_thw": g, "image_path": p}
                for g, p in zip(prompt_inputs["image_grid_thw"], images)
            ]
        else:
            prompt_inputs = processing_class(
                text=prompts_text,
                return_tensors=return_tensors,
                padding=padding,
                padding_side=padding_side,
                add_special_tokens=add_special_tokens,
            )
            additional_output = None       
    
        return prompt_inputs, additional_output

    
    @staticmethod
    def get_question_template(task_type: str):
        format_directive = (
            " Provide the positions of bounding boxes in the format: "
            "[[x1, y1, x2, y2], [x1, y1, x2, y2], ...]"
            "x1 and y1 for the top-left corner. "
            "x2 and y2 for the bottom-right corner. "
         )
        match task_type:
            case "rec":
                return f"{{Question}} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags. Output the final answer in JSON format.{format_directive}"
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
    def format_reward_rec(completions, **kwargs):
        import re
        import os
        from datetime import datetime
        # Updated pattern to match multiple bounding boxes
        pattern = r"<think>.*?</think>\s*<answer>\s*\[(\s*\[\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\]\s*,?\s*)+\]\s*</answer>"
        completion_contents = [completion[0]["content"] for completion in completions]
        matches = [re.search(pattern, content, re.DOTALL) is not None for content in completion_contents]
        
        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            with open(log_path.replace(".txt", "_format.txt"), "a", encoding='utf-8') as f:
                f.write(f"------------- {current_time} Format reward -------------\n")
                for content, match in zip(completion_contents, matches):
                    f.write(f"Content: {content}\n")
                    f.write(f"Has format: {bool(match)}\n")
        return [1.0 if match else 0.0 for match in matches]
    @staticmethod
    def iou_reward(completions, solution, **kwargs):
        """
        • Both GT and prediction boxes are **already in pixel coordinates**.  
        • Reward = Hungarian‑matched mean IoU                ∈ [0,1].  
        • If the *count* of predicted boxes equals the GT count, we add
          a small +0.1 bonus (capped at 1).  
        • Any parsing failure, shape error, or geometrically invalid box
          (x2 ≤ x1 or y2 ≤ y1) → reward = 0.
        """

        import re, os, json, numpy as np
        from scipy.optimize import linear_sum_assignment

        def parse_boxes(raw: str):
            raw = raw.strip()
            try:
                data = json.loads(raw)
            except Exception:
                nums = re.findall(r'-?\d+(?:\.\d+)?', raw)
                data = [float(n) for n in nums]
            if isinstance(data, list) and data and all(isinstance(x,(int,float)) for x in data):
                if len(data) % 4 != 0:
                    raise ValueError("count % 4 != 0")
                data = [data[i:i+4] for i in range(0, len(data), 4)]
            if not (isinstance(data, list) and all(isinstance(b, list) and len(b) == 4 for b in data)):
                raise ValueError("bad format")
            out = []
            for b in data:
                x1, y1, x2, y2 = map(float, b)
                if x2 <= x1 or y2 <= y1:
                    raise ValueError("degenerate box")
                out.append([x1, y1, x2, y2])
            return out

        def iou(a, b):
            ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
            ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            ua = (a[2]-a[0]) * (a[3]-a[1])
            ub = (b[2]-b[0]) * (b[3]-b[1])
            return inter / (ua + ub - inter) if ua + ub - inter else 0.0

        tag_pat  = r'<answer>(.*?)</answer>'
        pred_raw = [c[0]["content"] for c in completions]
        rewards  = []
        dbg  = str(os.getenv("DEBUG_MODE", "")).lower() in {"true", "1", "yes"}
        logf = os.getenv("LOG_PATH")
        def _d(msg: str):
            if dbg and logf:
                with open(logf, "a", encoding="utf-8") as fh:
                    fh.write(msg + "\n")
        for p_txt, s_txt in zip(pred_raw, solution):
            try:
                gt_boxes   = parse_boxes(re.findall(tag_pat, s_txt, re.DOTALL)[-1])
                pred_match = re.search(tag_pat, p_txt, re.DOTALL)
                if not pred_match:
                    raise ValueError("no <answer>")
                pred_boxes = parse_boxes(pred_match.group(1))
            except Exception:
                rewards.append(0.0)
                continue

            g, p = len(gt_boxes), len(pred_boxes)
            cost = np.zeros((g, p))
            for r, gt in enumerate(gt_boxes):
                for c, pr in enumerate(pred_boxes):
                    cost[r, c] = 1 - iou(gt, pr)          # Hungarian → minimise cost

            rows, cols = linear_sum_assignment(cost)
            
            if rows.size == 0 or cols.size == 0:          # nothing matched
                mean_iou = 0.0
            else:
                mean_iou = float(np.clip((1.0 - cost[rows, cols]).mean(), 0.0, 1.0))


            # ------------ sanitise ---------------------------------------
            if not np.isfinite(mean_iou):
                _d(f"[warn‑reward] mean_iou {mean_iou} → 0")
                mean_iou = 0.0
            else:                       # clamp to [0,1] just in case
                mean_iou = float(np.clip(mean_iou, 0.0, 1.0))

            bonus = 0.1 if g == p and g > 0 else 0.0
            reward  = min(mean_iou + bonus, 1.0)

            # final defence
            if not np.isfinite(reward):
                _d(f"[warn‑reward] reward {reward} → 0")
                reward = 0.0

            rewards.append(reward)

            # ---------- debug dump ---------------------------------------
            _d("-------")
            _d(f"IoU={mean_iou:.4f}  bonus={bonus:.2f}  reward={reward:.4f}")
            _d(f"GT   : {gt_boxes}")
            _d(f"Pred : {pred_boxes}")
            
        return rewards


    @staticmethod
    def select_reward_func(func: str, task_type: str):
        if func == "accuracy":
            if task_type == "rec":
                return Qwen2VLModule.iou_reward
            else:
                raise ValueError(f"Unsupported task type: {task_type}")
        elif func == "format":
            if task_type == "rec":
                return Qwen2VLModule.format_reward_rec
            else:
                raise ValueError(f"Unsupported task type: {task_type}")
        else:
            raise ValueError(f"Unsupported reward function: {func}")