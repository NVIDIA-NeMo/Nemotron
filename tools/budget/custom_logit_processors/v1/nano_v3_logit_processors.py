# copied from here https://docs.vllm.ai/en/v0.10.1.1/examples/offline_inference/logits_processor.html

from types import DynamicClassAttribute
from typing import Optional, Dict, Any, List

import torch
import json
import os

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.config import VllmConfig
from vllm.v1.sample.logits_processor import (
    BatchUpdate,
    LogitsProcessor,
    MoveDirectionality,
)
from vllm.v1.sample.logits_processor.builtin import process_dict_updates

#import os
#os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

NEWLINE_TOKENS = {12291, 20487, 8199, 57354, 126989, 14350, 71693, 2064, 26644, 10260, 114710, 10263, 106525, 81952, 10274, 51235, 94246, 28710, 79914, 86059, 2092, 79917, 18479, 18481, 96306, 32819, 2100, 14390, 67639, 122935, 110650, 2116, 124997, 12357, 104519, 67656, 114759, 106570, 114758, 86087, 4169, 100423, 102479, 12368, 14416, 14417, 38997, 34903, 67673, 102490, 26720, 94306, 12390, 61546, 53355, 32876, 98414, 90228, 26740, 127095, 39031, 18554, 47229, 6273, 51330, 4227, 2180, 100485, 98437, 112779, 96397, 108686, 43151, 110737, 4241, 69780, 49301, 86173, 118944, 4256, 18595, 16548, 102565, 12455, 114859, 92334, 104623, 6320, 118960, 118963, 55476, 67766, 14519, 55479, 123067, 61631, 32960, 2241, 114885, 63685, 116935, 125131, 63691, 104655, 57551, 73942, 100571, 16606, 73954, 16616, 114925, 14576, 20723, 4342, 67830, 14588, 123134, 33024, 4353, 39170, 26885, 123142, 55561, 28938, 37131, 84236, 12560, 43282, 6422, 24866, 90404, 14630, 45352, 20777, 31020, 20783, 14642, 65846, 8503, 4410, 100670, 53569, 2373, 20806, 6478, 104785, 41301, 115029, 20822, 8538, 6491, 47456, 94561, 24931, 72038, 117098, 31086, 43374, 80242, 43378, 110964, 61813, 14708, 92535, 53623, 82293, 39293, 94592, 104833, 72071, 98696, 84360, 10632, 55692, 22926, 27022, 31120, 117138, 74136, 123293, 106910, 37278, 86430, 37281, 125346, 68007, 65960, 88489, 98729, 43438, 33203, 108983, 57784, 102839, 49593, 100798, 68030, 92606, 16832, 14790, 43462, 37321, 4554, 86475, 96716, 14797, 43472, 109013, 29141, 2519, 12758, 14810, 16858, 72156, 10717, 61918, 125407, 80352, 84451, 12776, 92650, 80365, 6637, 125422, 104947, 8691, 92665, 23034, 14843, 12795, 14845, 72189, 14847, 107007, 66050, 80390, 64006, 12807, 4615, 123401, 74247, 14863, 94736, 25104, 90641, 37395, 2580, 35346, 84503, 2591, 102946, 76323, 125483, 18988, 70189, 74283, 121388, 14896, 98865, 100915, 90676, 35390, 80447, 98879, 47680, 100932, 86596, 27207, 2634, 43597, 2638, 107087, 4688, 2640, 121426, 82515, 88661, 8789, 8791, 21078, 127578, 84578, 82531, 19044, 21090, 94822, 33383, 115307, 25197, 62062, 64112, 10869, 72310, 8824, 14971, 17020, 43646, 96894, 2687, 129666, 86659, 27266, 119428, 12930, 8841, 123531, 29324, 2706, 113300, 111257, 31391, 82592, 37541, 15014, 49833, 2731, 96940, 68269, 2734, 115377, 94903, 21181, 62142, 31423, 86720, 6849, 45763, 43715, 76485, 72388, 92867, 17100, 51919, 49872, 60113, 58067, 117462, 27350, 41690, 10971, 56028, 53979, 10974, 53982, 31454, 39653, 78566, 127727, 31473, 60147, 35571, 64244, 35574, 97015, 119546, 31484, 84733, 2812, 127743, 45825, 2820, 21253, 101126, 27399, 78597, 97033, 11017, 29448, 86796, 6923, 27404, 117520, 84754, 2838, 35608, 76569, 125721, 8985, 82713, 70433, 49958, 29479, 82729, 2861, 60206, 37679, 66357, 113462, 2871, 74551, 119606, 19261, 13118, 97091, 74566, 113483, 6991, 47952, 35669, 11095, 105306, 95068, 13149, 2913, 35683, 11108, 9060, 66404, 123751, 23400, 107366, 33645, 127854, 58226, 9075, 47988, 25461, 80758, 21366, 107387, 62332, 115580, 103304, 31624, 25482, 68494, 54158, 50068, 93078, 11159, 13207, 80791, 115607, 56219, 97181, 117662, 21407, 84898, 33699, 41890, 129960, 78768, 25525, 19383, 19384, 119736, 74681, 113599, 58303, 19394, 95172, 119750, 5062, 109514, 60363, 15308, 95179, 43982, 121809, 76754, 23507, 9171, 62421, 80855, 23513, 76772, 74726, 27624, 111595, 1010, 60403, 7154, 80888, 44025, 29690, 97278, 7168, 11270, 5127, 62473, 58380, 54285, 76816, 107537, 3092, 56343, 39960, 115739, 3100, 44059, 19486, 44065, 89128, 105512, 23593, 109611, 11308, 68655, 117811, 87093, 121909, 7223, 9273, 44090, 7227, 107577, 66620, 130110, 3135, 11330, 9283, 80962, 99397, 97347, 23626, 25676, 123980, 95309, 29775, 115792, 50259, 68692, 87126, 27736, 42076, 19551, 19553, 58467, 103527, 130154, 50283, 66666, 99437, 19565, 44143, 95338, 99442, 117876, 40058, 111739, 50298, 11389, 11390, 68735, 87168, 7297, 7295, 66691, 97412, 126084, 58502, 23685, 19594, 23691, 101522, 109715, 19602, 11411, 25748, 130199, 130200, 52377, 40089, 115869, 17572, 9381, 107688, 76968, 79020, 33966, 111791, 50352, 13487, 58547, 93363, 91318, 54455, 56504, 52408, 5306, 83135, 70848, 25799, 11471, 38097, 21714, 60625, 79063, 91352, 111832, 5338, 40151, 77021, 46302, 9439, 50400, 3297, 3298, 25828, 52454, 23784, 107760, 36080, 1267, 3318, 17655, 118010, 15611, 120061, 111870, 70909, 93441, 87298, 60673, 115976, 124171, 7437, 7438, 48403, 66835, 58643, 34069, 64790, 109848, 5396, 46366, 52511, 21794, 60707, 83235, 44325, 105766, 21796, 93475, 23849, 46380, 27949, 103726, 19758, 68912, 56626, 19767, 40247, 1338, 5439, 73024, 87362, 25923, 54597, 7498, 1355, 116042, 9549, 9551, 5457, 56657, 70993, 81235, 34133, 1365, 19802, 34138, 109916, 52572, 32093, 120155, 89445, 73064, 11625, 56683, 109940, 21877, 71032, 56697, 3448, 103804, 111999, 71040, 60802, 73091, 48516, 62853, 17798, 73093, 7562, 109964, 32140, 85394, 36243, 118171, 21916, 13725, 60830, 7580, 71069, 118177, 13724, 114080, 3493, 128422, 73127, 21927, 3500, 1456, 120246, 21942, 1468, 36285, 32194, 3523, 36297, 50634, 19920, 15829, 93653, 30168, 128472, 3546, 69082, 114140, 17882, 101854, 1512, 11753, 9706, 3563, 46570, 60907, 13803, 1520, 36336, 22004, 69108, 124407, 48632, 95738, 85501, 52733, 75265, 99843, 30211, 87559, 122375, 24071, 48647, 32268, 128535, 1561, 48668, 11810, 32291, 1572, 83494, 95783, 38439, 114219, 89644, 120367, 101940, 75319, 3640, 52797, 91712, 1600, 28230, 22087, 3655, 38471, 20039, 46665, 34378, 65102, 75342, 38482, 116306, 1626, 34395, 7772, 120413, 73307, 30303, 83549, 54881, 28258, 83558, 13926, 1640, 1641, 104042, 106092, 11886, 28272, 56950, 65142, 99959, 24185, 106108, 128638, 59009, 11906, 26244, 65159, 24202, 56977, 61074, 34450, 24212, 102038, 93852, 81566, 46753, 18081, 79523, 85665, 69285, 9893, 50852, 104104, 61100, 7854, 28335, 128687, 61105, 102066, 26293, 24246, 38582, 81592, 3768, 57018, 61112, 93884, 22205, 126654, 124608, 89793, 38594, 28364, 130764, 20174, 63183, 14032, 50898, 9940, 28378, 46811, 102109, 57054, 124643, 83684, 55015, 44775, 3817, 79594, 48875, 67307, 3824, 110322, 57075, 67316, 16117, 20213, 9973, 120568, 12024, 130808, 69371, 100092, 104190, 65280, 38659, 12039, 1801, 28426, 77579, 53004, 61198, 32530, 108306, 59158, 3864, 30489, 130842, 102173, 87840, 5920, 1826, 69411, 87842, 16166, 24362, 3885, 128818, 32563, 1844, 89907, 53047, 108346, 69437, 44861, 71488, 69441, 89922, 94017, 100167, 1877, 110423, 106331, 53084, 63325, 65374, 100191, 42846, 20321, 38754, 106341, 118631, 100202, 116588, 112493, 46958, 36719, 14190, 71537, 102254, 89974, 104312, 12156, 38781, 10110, 3971, 40838, 112518, 65416, 36745, 87946, 57226, 10127, 92048, 3989, 34710, 118680, 20377, 71578, 24474, 38816, 1953, 71586, 24483, 6052, 83877, 128930, 110496, 67499, 122797, 28590, 126895, 20400, 102323, 63413, 30645, 10166, 92089, 114617, 49083, 120765, 57277, 88002, 79811, 40900, 55234, 38858, 65483, 104396, 83917, 4043, 57294, 38859, 2002, 26579, 120791, 30679, 88025, 57311, 26592, 30689, 12260, 100324, 53223, 71657, 4078, 100335, 28656, 32753, 22512, 8179, 4084, 67568, 2030, 79855, 61432, 122875, 116734}


class ThinkingBudgetLogitsProcessor(LogitsProcessor):
    def __init__(self, 
            vllm_config: VllmConfig, 
            device: torch.device, 
            is_pin_memory: bool):
        cfg_env = json.loads(os.getenv("THINKING_BUDGET_LOGITS_PROCESSOR_ARGS", "{}"))

        # Store a mapping from request index to output_tok_ids reference
        print("cfg_env in init:", cfg_env)
        self.thinking_budget = cfg_env.get("thinking_budget", -1)
        self.thinking_budget_grace_period = cfg_env.get("thinking_budget_grace_period", -1)
        self.end_token_ids  = cfg_env.get("end_token_ids", [])
        self.prompt_think_ids = cfg_env.get("prompt_think_ids", [])
        self.end_think_ids = cfg_env.get("end_think_ids", [])
        self.logit_processor_state: dict[int, dict[Any, Any]] = {}

    def is_argmax_invariant(self) -> bool:
        return False  # This processor does not affect sampling

    def update_state(self, batch_update: Optional[BatchUpdate]):
        if not batch_update:
            return
        # Add new requests
        for index, sampling_params, prompt_tok_ids, output_tok_ids in batch_update.added:
            state = self.logit_processor_state.get(index, {})
            state["output_tok_ids"] = output_tok_ids
            state["thinking_budget"] = self.thinking_budget
            state["thinking_budget_grace_period"] = self.thinking_budget_grace_period
            state["end_token_ids"] = self.end_token_ids
            state["is_thinking"] = False
            
            if sampling_params.extra_args:
                """
                sampling params can overwrite ones from the cfg_env
                """
                state["thinking_budget"] = sampling_params.extra_args.get("thinking_budget", self.thinking_budget)
                state["thinking_budget_grace_period"] = sampling_params.extra_args.get("thinking_budget_grace_period", self.thinking_budget_grace_period)
                state["end_token_ids"] = json.loads(sampling_params.extra_args["end_token_ids"]) if "end_token_ids" in sampling_params.extra_args else self.end_token_ids

            if prompt_tok_ids[-len(self.prompt_think_ids):] == self.prompt_think_ids:  # check for \n<think>\n at the end of the prompt which indicates that the model is in thinking mode.
                print("model starting thinking...")
                state["is_thinking"] = True
                state["start_of_end"] = False
                state["end_of_end"] = False
            self.logit_processor_state[index] = state
        # Remove finished requests
        for index in batch_update.removed:
            self.logit_processor_state.pop(index, None)
        # Handle moved requests
        for a, b, direction in batch_update.moved:
            a_val = self.logit_processor_state.pop(a, None)
            b_val = self.logit_processor_state.pop(b, None)
            if a_val is not None:
                self.logit_processor_state[b] = a_val
            if direction.name == "SWAP" and b_val is not None:
                self.logit_processor_state[a] = b_val
    
    def _suffix_prefix_overlap(self, a, b):
        m = min(len(a), len(b))
        for k in range(m, 0, -1):           # try longest first
            if a[-k:] == b[:k]:
                return k
        return 0
    
    def _maybe_end_thinking(self, idx: int, logits: torch.Tensor, state: Dict[Any, Any]):
        if state["end_of_end"]:
            return logits
        
        for eti in self.end_think_ids:
            check_if_think_ended_naturally = list(state["output_tok_ids"][-len(eti):])
            if check_if_think_ended_naturally == eti:
                # if thinking ends normally don't intervene...
                state["start_of_end"] = True
                state["end_of_end"] = True


        if len(state["output_tok_ids"]) >= state["thinking_budget"] + state["thinking_budget_grace_period"] and not state["start_of_end"]:
            state["start_of_end"] = True

        if len(state["output_tok_ids"]) >= state["thinking_budget"] and state["output_tok_ids"][-1] in NEWLINE_TOKENS and not state["start_of_end"]:
            state["start_of_end"] = True
        
        if state["start_of_end"] and not state["end_of_end"]:
            end_token_ids = state["end_token_ids"]
            last_n_inputs = list(state["output_tok_ids"][-len(end_token_ids):])
            overlap = self._suffix_prefix_overlap(last_n_inputs, end_token_ids)
            if overlap < len(end_token_ids):
                logits[idx, :] = float("-inf") 
                insert_id = end_token_ids[overlap]
                logits[idx, insert_id] = 1.0
            else:
                state["end_of_end"] = True
        return logits

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        for idx, state in self.logit_processor_state.items():
            if state.get("is_thinking", False):
                logits = self._maybe_end_thinking(idx, logits, state)
        return logits

def main():
    model = "nvidia/NVIDIA-Nemotron-Nano-31B-A3-v3"
    msg = """Bob is an avid fan of the video game \"League of Leesins\", and today he celebrates as the League of Leesins World Championship comes to an end! \n\nThe tournament consisted of $n$ ($n \\ge 5$) teams around the world. Before the tournament starts, Bob has made a prediction of the rankings of each team, from $1$-st to $n$-th. After the final, he compared the prediction with the actual result and found out that the $i$-th team according to his prediction ended up at the $p_i$-th position ($1 \\le p_i \\le n$, all $p_i$ are unique). In other words, $p$ is a permutation of $1, 2, \\dots, n$.\n\nAs Bob's favorite League player is the famous \"3ga\", he decided to write down every $3$ consecutive elements of the permutation $p$. Formally, Bob created an array $q$ of $n-2$ triples, where $q_i = (p_i, p_{i+1}, p_{i+2})$ for each $1 \\le i \\le n-2$. Bob was very proud of his array, so he showed it to his friend Alice.\n\nAfter learning of Bob's array, Alice declared that she could retrieve the permutation $p$ even if Bob rearranges the elements of $q$ and the elements within each triple. Of course, Bob did not believe in such magic, so he did just the same as above to see Alice's respond.\n\nFor example, if $n = 5$ and $p = [1, 4, 2, 3, 5]$, then the original array $q$ will be $[(1, 4, 2), (4, 2, 3), (2, 3, 5)]$. Bob can then rearrange the numbers within each triple and the positions of the triples to get $[(4, 3, 2), (2, 3, 5), (4, 1, 2)]$. Note that $[(1, 4, 2), (4, 2, 2), (3, 3, 5)]$ is not a valid rearrangement of $q$, as Bob is not allowed to swap numbers belong to different triples.\n\nAs Alice's friend, you know for sure that Alice was just trying to show off, so you decided to save her some face by giving her any permutation $p$ that is consistent with the array $q$ she was given. \n\n\n-----Input-----\n\nThe first line contains a single integer $n$ ($5 \\le n \\le 10^5$) — the size of permutation $p$.\n\nThe $i$-th of the next $n-2$ lines contains $3$ integers $q_{i, 1}$, $q_{i, 2}$, $q_{i, 3}$ ($1 \\le q_{i, j} \\le n$) — the elements of the $i$-th triple of the rearranged (shuffled) array $q_i$, in random order. Remember, that the numbers within each triple can be rearranged and also the positions of the triples can be rearranged.\n\nIt is guaranteed that there is at least one permutation $p$ that is consistent with the input. \n\n\n-----Output-----\n\nPrint $n$ distinct integers $p_1, p_2, \\ldots, p_n$ ($1 \\le p_i \\le n$) such that $p$ is consistent with array $q$. \n\nIf there are multiple answers, print any. \n\n\n-----Example-----\nInput\n5\n4 3 2\n2 3 5\n4 1 2\n\nOutput\n1 4 2 3 5\n\n\nRead the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows. Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT.\n```python\n# YOUR CODE HERE\n```"""
    messages = [{"role": "system", "content": "You are a helpful assistant. /think"},{"role": "user", "content": msg}]
    messages2= [{"role": "system", "content": "You are a helpful assistant. /think"},{"role": "user", "content": "Write a haiku about a cat"}]
    tokenizer = AutoTokenizer.from_pretrained(model)

    prompts = [tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, add_special_tokens=False),
            tokenizer.apply_chat_template(messages2, tokenize=False, add_generation_prompt=True, add_special_tokens=False)]

    sampling_params_list = [SamplingParams(temperature=0.6, max_tokens=1220, extra_args={"thinking_budget": 150, "thinking_budget_grace_period": 30, "end_token_ids":[1871, 5565, 11483, 6139, 1046, 2259, 74045, 1062]}), # Reached thinking limit. </think>
                            SamplingParams(temperature=0.6, max_tokens=1260, extra_args={"thinking_budget": 120, "thinking_budget_grace_period": 20, "end_token_ids":[2259, 74045, 1062]})] # </think>

    llm = LLM(
            model=model,
            logits_processors=[ThinkingBudgetLogitsProcessor],
            trust_remote_code=True,
            )
    outputs = llm.generate(prompts, sampling_params_list)
    print("\nGenerated Outputs:\n" + "-" * 60)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt:    {prompt!r}")
        print(f"Output:    {generated_text!r}")
        print("-" * 60)

if __name__ == "__main__":
    main()
