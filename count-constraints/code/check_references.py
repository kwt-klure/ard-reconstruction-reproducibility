"""Independently verify all 200 conditional references, integer supports and masks."""
from concurrent.futures import ProcessPoolExecutor
import argparse
import json
from pathlib import Path
from independent import verify_graph
from plugin_readout import routes

def check(case):
    result=verify_graph(case["input"],case["reference"])
    return {"cohort":case["cohort"],"seed":case["seed"],
            **{k:v for k,v in result["errors"].items() if k.startswith("max_abs_")}}

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--workers",type=int,default=2)
    args=parser.parse_args()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results=list(pool.map(check,routes()))
    with args.output.open("x") as stream:
        json.dump(results,stream,indent=2)
    print(f"Independently checked {len(results)} conditional references")
