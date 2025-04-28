# metta/sim/replay_recorder.py

import json
import os
import zlib
import boto3
import wandb
import logging

logger = logging.getLogger(__name__)

class ReplayRecorder:
    """Records and saves a single replay."""
    
    def __init__(self, replay_path, wandb_run=None):
        self._replay_path = replay_path
        self._wandb_run = wandb_run
        self._replay_data = None
        
        # S3 configuration
        self._s3_client = boto3.client("s3")
        
    def initialize(self, env):
        """Initialize replay data structure for a single environment."""
        self._grid_objects = []
        
        self._replay_data = {
            "version": 1,
            "action_names": env._c_env.action_names(),
            "object_types": env._c_env.object_type_names(),
            "map_size": [env._c_env.map_width, env._c_env.map_height],
            "num_agents": env._c_env.num_agents,
            "max_steps": 0,
            "grid_objects": self._grid_objects,
        }
    
    def _add_sequence_key(self, grid_object, key, step, value):
        """Add a key to the replay that is a sequence of values."""
        if key not in grid_object:
            # Add new key.
            grid_object[key] = [[step, value]]
        else:
            # Only add new entry if it has changed:
            if grid_object[key][-1][1] != value:
                grid_object[key].append([step, value])
    
    def update(self, step, env, actions, rewards, total_rewards, action_success, agent_offset=0):
        """Update replay data for the current step."""
        if not self._replay_data:
            return
            
        actions_array = actions.cpu().numpy()
        
        for i, grid_object in enumerate(env.grid_objects.values()):
            if len(self._grid_objects) <= i:
                # Add new grid object.
                self._grid_objects.append({})
            
            for key, value in grid_object.items():
                self._add_sequence_key(self._grid_objects[i], key, step, value)
            
            if "agent_id" in grid_object:
                agent_id = grid_object["agent_id"]
                global_agent_id = agent_id + agent_offset
                
                self._add_sequence_key(self._grid_objects[i], "action", step, actions_array[global_agent_id].tolist())
                self._add_sequence_key(
                    self._grid_objects[i], "action_success", step, bool(action_success[agent_id])
                )
                self._add_sequence_key(self._grid_objects[i], "reward", step, rewards[global_agent_id].item())
                self._add_sequence_key(
                    self._grid_objects[i], "total_reward", step, total_rewards[global_agent_id].item()
                )
        
        self._replay_data["max_steps"] = step + 1
    
    def save(self):
        """Save replay data to file and optionally upload to S3."""
        if not self._replay_data:
            return
            
        # Trim value changes to make them more compact.
        for grid_object in self._grid_objects:
            for key, changes in grid_object.items():
                if len(changes) == 1:
                    grid_object[key] = changes[0][1]
        
        local_path = self._replay_path
        
        # For S3 paths, use a temp local file
        upload_to_s3 = False
        if self._replay_path.startswith("s3://"):
            upload_to_s3 = True
            s3_path = self._replay_path
            temp_dir = os.path.join("/tmp", "metta_replays")
            os.makedirs(temp_dir, exist_ok=True)
            filename = os.path.basename(local_path)
            local_path = os.path.join(temp_dir, filename)
        
        # Make sure path has proper extension
        if not local_path.endswith(".json.z"):
            local_path = local_path + ".json.z"
        
        # Make sure the directory exists
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        # Compress it with deflate.
        replay_data = json.dumps(self._replay_data)
        replay_bytes = replay_data.encode("utf-8")
        compressed_data = zlib.compress(replay_bytes)
        
        with open(local_path, "wb") as f:
            f.write(compressed_data)
        
        logger.info(f"Replay saved to {local_path}")
        
        # Upload to S3 if configured
        if upload_to_s3:
            self._upload_to_s3(local_path, s3_path)
    
    def _upload_to_s3(self, local_path, s3_path):
        """Upload the replay to S3 and log the link to WandB."""
        # Parse S3 path
        bucket_and_key = s3_path.split("s3://")[1]
        parts = bucket_and_key.split("/", 1)
        
        s3_bucket = parts[0]
        if len(parts) > 1:
            s3_key = parts[1]
        else:
            s3_key = os.path.basename(local_path)
            
        # Upload
        self._s3_client.upload_file(
            Filename=local_path, 
            Bucket=s3_bucket, 
            Key=s3_key, 
            ExtraArgs={"ContentType": "application/x-compress"}
        )
        
        logger.info(f"Uploaded replay to s3://{s3_bucket}/{s3_key}")
        
        # Log link to WandB
        if self._wandb_run:
            link = f"https://{s3_bucket}.s3.us-east-1.amazonaws.com/{s3_key}"
            player_url = f"https://metta-ai.github.io/metta/?replayUrl={link}"
            
            display_name = os.path.basename(self._replay_path)
            epoch = display_name.split(".")[-2] if "." in display_name else ""
            
            link_summary = {
                "replays/link": wandb.Html(f'<a href="{player_url}">MetaScope Replay (Epoch {epoch})</a>')
            }
            self._wandb_run.log(link_summary)