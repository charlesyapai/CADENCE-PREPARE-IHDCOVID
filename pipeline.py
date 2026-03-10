
import yaml
import sys
import os
import argparse
from importlib import import_module

# Add local directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def load_config(path=None):
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="IHD-COVID Analysis Pipeline")
    parser.add_argument("--step", type=int, help="Run specific step (1, 2, etc). If 0 or omitted, run all.")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    
    args = parser.parse_args()
    config = load_config(args.config)
    
    print("="*60)
    print(f" PIPELINE STARTING | Project: {config['project_name']}")
    print("="*60)
    
    # Define Steps
    # We import dynamically or just mapped
    # 1: extract_index_events
    # 2: generate_cohorts
    
    steps = {
        1: "1_extract_index_events",
        2: "2_generate_cohorts",
        3: "3_enrich_features",
        4: "4_statistical_analysis",
        5: "5_variant_era_analysis",
        6: "6_tier_1_analysis",
        7: "7_cci_diagcode_discovery",
        8: "8_apply_cci_codes",
        9: "9_tier_2_analysis",
    }
    
    # Map step numbers to their entry-point function names
    # (Most follow run_step_N, but newer scripts may differ)
    func_names = {
        1: "run_step_1",
        2: "run_step_2",
        3: "run_step_3",
        4: "run_step_4",
        5: "run_step_5",
        6: "run_tier_1",
        7: "run_step_7",
        8: "run_step_8",
        9: "run_step_9",
    }
    
    to_run = []
    if args.step:
        if args.step in steps:
            to_run.append(args.step)
        else:
            print(f"Unknown step: {args.step}")
            sys.exit(1)
    else:
        to_run = sorted(steps.keys())
        
    for step_num in to_run:
        module_name = steps[step_num]
        print(f"\n>>> EXECUTING STEP {step_num}: {module_name}")
        
        try:
            # Dynamic import
            mod = import_module(module_name)
            
            func_name = func_names.get(step_num, f"run_step_{step_num}")
            if hasattr(mod, func_name):
                func = getattr(mod, func_name)
                func(config)
                print(f">>> STEP {step_num} COMPLETE.")
            else:
                print(f"[ERROR] Module {module_name} missing function {func_name}")
                
        except Exception as e:
            print(f"[FATAL] Pipeline failed at Step {step_num}: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
            
    print("\n" + "="*60)
    print(" PIPELINE EXECUTION FINISHED")
    print("="*60)

if __name__ == "__main__":
    main()
