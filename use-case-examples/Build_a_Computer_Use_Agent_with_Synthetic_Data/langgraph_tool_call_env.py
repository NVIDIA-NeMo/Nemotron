from typing import TypedDict, Optional, List, Dict, Any
import json

import ray
import torch
from nemo_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn

class LangGraphCLIEnvMetadata(TypedDict):
    """Metadata for a LangGraph CLI command environment."""
    reference_command: str               # expected CLI command name
    reference_flags: Dict[str, Any]      # expected flags (key-value pairs) for the command


class LangGraphCLIEnvConfig(TypedDict, total=False):
    stop_strings: Optional[List[str]]    # optional stop tokens for generation


def score_cli_output(predicted_json: dict, reference_json: dict) -> float:
    """Compare predicted CLI JSON vs reference, returning a reward."""
    # Extract the expected command and flags
    ref_cmd = reference_json.get("command")
    ref_flags = reference_json.get("flags", {})
    pred_cmd = predicted_json.get("command")
    pred_flags = predicted_json.get("flags", {})
    
    # If command is wrong, give a full penalty (no partial credit)
    if pred_cmd != ref_cmd:
        return -1.0  # wrong tool invoked
    
    # If command is correct, evaluate flags
    total_flags = len(ref_flags)
    correct_count = 0
    wrong_value_count = 0
    # Count correctly predicted flags (matching key and value)
    for key, ref_val in ref_flags.items():
        if key in pred_flags:
            if pred_flags[key] == ref_val:
                correct_count += 1
            else:
                # key present but value incorrect
                wrong_value_count += 1
    # Count hallucinated flags (in prediction but not in reference)
    extra_count = sum(1 for k in pred_flags.keys() if k not in ref_flags)
    
    # Compute partial reward
    # Fraction of correctly provided flags minus fractions for wrong/extra flags
    reward = (correct_count - wrong_value_count - extra_count) / total_flags if total_flags > 0 else 0.0
    # Cap the reward between -1.0 and 1.0
    if reward > 1.0:
        reward = 1.0
    if reward < -1.0:
        reward = -1.0
    # If all flags correct and no extras, ensure reward is exactly 1.0
    if correct_count == total_flags and wrong_value_count == 0 and extra_count == 0:
        reward = 1.0
    return reward

class LangGraphCLIRunner:
    """Processes a single turn in the LangGraph CLI environment."""
    def __init__(self):
        # No persistent state needed for single-turn evaluation
        pass
    
    def _parse_model_output(self, message_log: List[Dict[str, str]]) -> Optional[dict]:
        """Extract and parse the assistant's JSON response from the message log."""
        # The assistant's last message content is the JSON string output
        assistant_messages = [m["content"] for m in message_log if m["role"] == "assistant"]
        if not assistant_messages:
            return None
        output_str = "".join(assistant_messages)  # combine in case of multiple assistant messages
        try:
            return json.loads(output_str)
        except Exception:
            return None  # return None if JSON parsing fails
    
    def process_turn(self, message_log: List[Dict[str, str]], metadata: LangGraphCLIEnvMetadata
    ) -> tuple[Dict[str, str], float, bool, Optional[List[str]], Optional[LangGraphCLIEnvMetadata]]:
        """Process the model's output for one turn and return (observation, reward, terminated, stop_strings, new_metadata)."""
        # Parse model output JSON
        pred_json = self._parse_model_output(message_log)
        # Reference JSON (for scoring)
        reference_json = {
            "command": metadata["reference_command"],
            "flags": metadata["reference_flags"]
        }
        # Initialize variables for reward and environment response
        terminated = True  # one-turn episode ends after this evaluation
        stop_strings = None  # no further stop strings needed (episode ends)
        
        if pred_json is None:
            # Failed to produce valid JSON -> heavy penalty
            reward = -1.0
            feedback = "Invalid JSON output."
            exact_match = False
            command_correct = False
            flag_accuracy = 0.0
        else:
            # Compute reward using the scoring logic
            reward = score_cli_output(pred_json, reference_json)
            # Determine feedback message and metrics
            # Check if command was correct and if flags exactly match
            command_correct = (pred_json.get("command") == metadata["reference_command"])
            exact_match = (command_correct and reward == 1.0)
            # Calculate fraction of flags correctly provided (flag accuracy)
            total_flags = len(metadata["reference_flags"])
            if total_flags > 0 and command_correct:
                # Recompute correct flag fraction for logging (ensure non-negative)
                correct_flags = sum(
                    1 for k, v in metadata["reference_flags"].items()
                    if k in pred_json.get("flags", {}) and pred_json["flags"][k] == v
                )
                flag_accuracy = correct_flags / total_flags
            else:
                # If command wrong or no flags expected
                flag_accuracy = 0.0
            
            # Create an environment feedback message based on the outcome
            if exact_match:
                feedback = "Exact match! Command and all flags are correct."
            elif command_correct:
                # Command correct but not perfect
                if reward > 0:
                    feedback = "Partial match: correct command, some flags correct."
                else:
                    feedback = "Command correct, but flags are incorrect or extra."
            else:
                feedback = "Incorrect command."
        
        # Prepare the observation from the environment (environment's message to the model)
        observation = {
            "role": "environment",
            "content": feedback
        }
        # Package updated metadata with results (could be used for metrics)
        new_metadata = {
            "exact_match": exact_match,
            "command_correct": command_correct,
            "flag_accuracy": flag_accuracy,
            # We can include the original reference too if needed, or drop it
            "reference_command": metadata["reference_command"],
            "reference_flags": metadata["reference_flags"]
        }
        return observation, reward, terminated, stop_strings, new_metadata


@ray.remote
class LangGraphCLIEnv(EnvironmentInterface):
    """NeMo-RL environment for LangGraph CLI command generation and evaluation."""
    DEFAULT_STOP_STRINGS = ['}']  # default stop at end of JSON object
    
    def __init__(self, cfg: Optional[LangGraphCLIEnvConfig] = None):
        # Use provided config or defaults
        self.cfg = cfg or {}
        # Determine stop strings for the environment (if any)
        self.stop_strings = self.cfg.get("stop_strings", self.DEFAULT_STOP_STRINGS)
        # Initialize a runner instance
        self.runner = LangGraphCLIRunner()
    
    def step(self, message_log_batch: List[List[Dict[str, str]]],
             metadata_batch: List[LangGraphCLIEnvMetadata]
    ) -> EnvironmentReturn:
        """Process a batch of model outputs through the environment."""
        batch_size = len(message_log_batch)
        observations: List[Dict[str, str]] = []
        new_metadata_list: List[Optional[dict]] = []
        rewards_list: List[float] = []
        terminated_list: List[bool] = []
        next_stop_list: List[Optional[List[str]]] = []
        
        # Process each sample in the batch
        for i in range(batch_size):
            message_log = message_log_batch[i]
            meta = metadata_batch[i]
            # Run the single-turn evaluation
            obs, reward, terminated, stop_strs, updated_meta = self.runner.process_turn(message_log, meta)
            observations.append(obs)
            new_metadata_list.append(updated_meta)  # updated metadata per sample
            rewards_list.append(reward)
            terminated_list.append(terminated)
            # For next stop strings, we can either provide environment-defined stop or None if episode ended
            next_stop_list.append(stop_strs if stop_strs is not None else None)
        
        # Convert rewards and terminated lists to torch Tensors (as required by EnvironmentReturn)
        rewards_tensor = torch.tensor(rewards_list, dtype=torch.float32)
        terminateds_tensor = torch.tensor(terminated_list, dtype=torch.bool)
        
        # Return the standard EnvironmentReturn tuple:contentReference[oaicite:12]{index=12}:contentReference[oaicite:13]{index=13}.
        return EnvironmentReturn(
            observations=observations,
            metadata=new_metadata_list,
            next_stop_strings=next_stop_list,
            rewards=rewards_tensor,
            terminateds=terminateds_tensor
        )
    
    def global_post_process_and_metrics(self, batch: "BatchedDataDict"
    ) -> tuple["BatchedDataDict", dict]:
        """
        Compute global metrics after all rollouts. 
        Returns the (possibly modified) batch and a metrics dictionary.
        """
        # Extract the list of final metadata for each sample (each should include our flags from updated_meta)
        final_metadata_list: List[Optional[dict]] = batch["metadata"]
        total_samples = len(final_metadata_list)
        exact_match_count = 0
        command_correct_count = 0
        total_flag_accuracy = 0.0
        
        for meta in final_metadata_list:
            if meta is None:
                continue  # safety check
            # Check if the command was correct for this sample
            if meta.get("command_correct"):
                command_correct_count += 1
            if meta.get("exact_match"):
                exact_match_count += 1
            # Sum up flag accuracy (ensure we have a float value)
            total_flag_accuracy += float(meta.get("flag_accuracy", 0.0))
        
        # Calculate metrics
        exact_match_rate = exact_match_count / total_samples if total_samples > 0 else 0.0
        # Partial match = cases where command was correct but not an exact match
        partial_match_count = command_correct_count - exact_match_count
        partial_match_rate = partial_match_count / total_samples if total_samples > 0 else 0.0
        average_flag_accuracy = total_flag_accuracy / total_samples if total_samples > 0 else 0.0
        
        metrics = {
            "exact_match_rate": exact_match_rate,
            "partial_match_rate": partial_match_rate,
            "average_flag_accuracy": average_flag_accuracy
        }
        return batch, metrics