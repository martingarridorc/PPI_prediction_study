import wandb
import sys
import os
import argparse
import wandb
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main import train

if False:
    sweep_id = "ar82d8ts"
    wandb.agent(sweep_id, function=train(sweep=True), count=1)
else:

    parser = argparse.ArgumentParser(description='Start a wandb agent with a sweep ID')
    parser.add_argument('--sweep_id', type=str, help='Sweep ID from wandb')
    args = parser.parse_args()

    sweep_id = args.sweep_id
    wandb.agent(sweep_id, function=train, count=5, project="hypertune")