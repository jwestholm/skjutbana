from __future__ import annotations
import argparse,json
from pathlib import Path
from src.engine.ai.new_hole_ai_v217 import NewHoleAIV217

def main() -> int:
    p=argparse.ArgumentParser(description='Verify saved V2.17 NEW-hole model/report'); p.add_argument('--model',type=Path,default=Path('content/ai/reports/v217/new_hole_ai_v217.npz')); p.add_argument('--report',type=Path,default=Path('content/ai/reports/v217/new_hole_v217_report.json')); args=p.parse_args()
    model,meta=NewHoleAIV217.load(args.model); report=json.loads(args.report.read_text(encoding='utf-8'))
    print('V2.17 NEW-HOLE VERIFY'); print('====================='); print(f'Model               : {args.model}'); print(f'Input dim           : {model.config.input_dim}'); print(f"Threshold           : {meta.get('metadata',{}).get('threshold')}"); print(f"Shadow only         : {meta.get('metadata',{}).get('shadow_only')}"); print(f"Split provisional   : {report.get('split_is_provisional')}"); print(f"Sessions            : {report.get('dataset',{}).get('sessions')}"); print(f"Live authority      : {report.get('gate',{}).get('eligible_for_live_authority')}")
    contract=meta.get('metadata',{}).get('semantic_contract') or {}; print(f"Positive semantic   : {contract.get('positive')}"); print(f"Negative semantic   : {contract.get('negative')}")
    if bool(meta.get('metadata',{}).get('shadow_only')) is not True: raise SystemExit('ERROR: V2.17 model is not marked shadow_only')
    if report.get('gate',{}).get('eligible_for_live_authority') is not False: raise SystemExit('ERROR: V2.17 report unexpectedly grants live authority')
    print('Verified. V2.17 cannot grant live authority.'); return 0
if __name__=='__main__': raise SystemExit(main())
