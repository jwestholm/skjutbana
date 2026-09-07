from __future__ import annotations
import json
from pathlib import Path
from typing import Any
ROOT=Path(__file__).resolve().parents[1]
MENU_PATH=ROOT/'content/menu.json'; ENTRY_PATH=ROOT/'menu_games_entry_v253.json'; ENTRY_ID='game_objects_test_v250'

def _walk(node: Any):
    if isinstance(node,dict):
        yield node
        for v in node.values(): yield from _walk(v)
    elif isinstance(node,list):
        for v in node: yield from _walk(v)

def patch_menu_text(text:str,entry:dict[str,Any]):
    data=json.loads(text); found=any(str(n.get('id',''))==ENTRY_ID for n in _walk(data))
    if not found: return text,False,False
    marker=f'"id": "{ENTRY_ID}"'; idx=text.find(marker)
    if idx<0: marker=f'"id":"{ENTRY_ID}"'; idx=text.find(marker)
    if idx<0:
        changed=False
        for n in _walk(data):
            if str(n.get('id',''))==ENTRY_ID:
                for k,v in entry.items():
                    if n.get(k)!=v: n[k]=v; changed=True
                break
        return json.dumps(data,ensure_ascii=False,indent=2)+'\n',changed,True
    start=text.rfind('{',0,idx); depth=0; ins=False; esc=False; end=None
    for pos in range(start,len(text)):
        ch=text[pos]
        if ins:
            if esc: esc=False
            elif ch=='\\': esc=True
            elif ch=='"': ins=False
            continue
        if ch=='"': ins=True
        elif ch=='{': depth+=1
        elif ch=='}':
            depth-=1
            if depth==0: end=pos+1; break
    obj_text=text[start:end]; obj=json.loads(obj_text); changed=False
    for k in ('title','description','script'):
        if k in entry and obj.get(k)!=entry[k]:
            old=json.dumps(obj.get(k),ensure_ascii=False); new=json.dumps(entry[k],ensure_ascii=False)
            needle=f'"{k}": {old}'
            if needle not in obj_text: needle=f'"{k}":{old}'
            if needle in obj_text:
                obj_text=obj_text.replace(needle,needle.split(old)[0]+new,1); obj[k]=entry[k]; changed=True
    patched=text[:start]+obj_text+text[end:]; json.loads(patched); return patched,changed,True

def main():
    try:
        from automation.v250_apply_menu import main as prev
        prev()
    except Exception as exc: print(f'[WARN] V2.25.0 menu preparation could not run: {exc}')
    entry=json.loads(ENTRY_PATH.read_text(encoding='utf-8')); text=MENU_PATH.read_text(encoding='utf-8')
    patched,changed,found=patch_menu_text(text,entry)
    if not found: raise RuntimeError(f'Could not find {ENTRY_ID}')
    if changed: MENU_PATH.write_text(patched,encoding='utf-8'); print('[PATCH] content/menu.json -> Game Objects Test (V2.25.3)')
    else: print('[OK] content/menu.json already labels Game Objects Test (V2.25.3)')
if __name__=='__main__': main()
