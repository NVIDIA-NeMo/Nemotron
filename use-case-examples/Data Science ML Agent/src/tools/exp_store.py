# exp_store.py - Simplified for beginners
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

class ExperimentStore:
    """Simple storage for machine learning experiment results."""
    
    def __init__(self, runs_file="runs/experiments.jsonl"):
        self.runs_file = Path(runs_file)
        
        # Create the directory if it doesn't exist
        self.runs_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Create the file if it doesn't exist
        if not self.runs_file.exists():
            self.runs_file.touch()

    def clear_all(self):
        """Delete all stored experiments."""
        self.runs_file.write_text("", encoding="utf-8")

    # Persist results
    def save_experiment(self, experiment_data: Dict):
        """Save an experiment result."""
        # Add timestamp
        experiment_data["timestamp"] = time.time()
        experiment_data["date"] = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Append to file
        with self.runs_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(experiment_data) + "\n")

    def load_all_experiments(self) -> List[Dict]:
        """Load all saved experiments."""
        experiments = []
        
        if not self.runs_file.exists():
            return experiments
        
        with self.runs_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    try:
                        experiment = json.loads(line)
                        experiments.append(experiment)
                    except json.JSONDecodeError:
                        # Skip corrupted lines
                        continue
        
        return experiments

    def find_best_experiment(self, metric_name: str) -> Optional[Dict]:
        """Find the experiment with the best score for a given metric."""
        experiments = self.load_all_experiments()
        
        if not experiments:
            return None
        
        best_experiment = None
        best_score = None
        
        # Determine if higher or lower scores are better
        higher_is_better = metric_name.lower() in [
            'accuracy', 'f1', 'f1_weighted', 'r2', 'precision', 'recall'
        ]
        
        for experiment in experiments:
            # Look for the metric in the experiment's metrics
            metrics = experiment.get('metrics', {})
            score = metrics.get(metric_name)
            
            if score is None:
                continue
                
            if best_experiment is None:
                best_experiment = experiment
                best_score = score
            else:
                if higher_is_better and score > best_score:
                    best_experiment = experiment
                    best_score = score
                elif not higher_is_better and score < best_score:
                    best_experiment = experiment
                    best_score = score
        
        return best_experiment

    def get_recent_experiments(self, limit: int = 10) -> List[Dict]:
        """Get the most recent experiments."""
        experiments = self.load_all_experiments()
        
        # Sort by timestamp (most recent first)
        experiments.sort(key=lambda x: x.get('timestamp', 0), reverse=True)
        
        return experiments[:limit]

    def count_experiments(self) -> int:
        """Count total number of experiments."""
        return len(self.load_all_experiments())