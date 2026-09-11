"""One-command offline inventory/quality/dataset rebuild orchestration."""
from __future__ import annotations
import argparse,subprocess,sys
from pathlib import Path
def main(root,out):
 out.mkdir(parents=True,exist_ok=True)
 commands=[['-m','automation.physical_data_inventory','--root',str(root),'--output',str(out/'inventory.json')],['-m','automation.physical_trace_quality','--root',str(root),'--output',str(out/'quality.json')],['-m','automation.physical_patch_dataset','--root',str(root),'--output',str(out/'patch_dataset.json')]]
 for c in commands: subprocess.run([sys.executable,*c],check=True)
 print(out)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('content/ai/physical_traces'));p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.root,a.output)
