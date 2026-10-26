"""Summarise saved per-frame counts without loading a network or raw imagery."""
from .common import ROOT,read_csv,write_csv
from collections import defaultdict
import statistics,json

def summarise(rows):
    out=[]
    for run in dict.fromkeys(r['run'] for r in rows):
        rr=[r for r in rows if r['run']==run];by=defaultdict(list)
        for r in rr:by[r['group']].append(r)
        record=dict(run=run,model=rr[0]['model'],seed=int(rr[0]['seed']))
        for metric in ['Dice','Recall','precision','AP']:
            values=[statistics.mean(float(r[metric]) for r in gg if int(r['FG'])) for gg in by.values() if any(int(r['FG']) for r in gg)]
            record[metric]=statistics.mean(values) if values else None
        for name,part in [('anomalous',[r for r in rr if int(r['FG'])]),('zero_region',[r for r in rr if not int(r['FG'])])]:
            fp=sum(int(r['FP']) for r in part);bg=sum(int(r['BG']) for r in part)
            record[name+'_FP']=fp;record[name+'_BG']=bg;record[name+'_background_FP']=fp/bg if bg else None
        for c in range(1,8):
            tp=sum(int(r[f'class{c}_TP']) for r in rr);n=sum(int(r[f'class{c}_pixels']) for r in rr)
            record[f'class{c}_Recall']=tp/n if n else None
        out.append(record)
    return out

def main():
    rows=[]
    for path in sorted((ROOT/'results').glob('*_holdout_per_frame.csv')):rows.extend(read_csv(path))
    if not rows:raise FileNotFoundError('No saved holdout per-frame counts')
    out=summarise(rows);write_csv(ROOT/'results/reproduced_summary.csv',out)
    print(json.dumps(out,indent=2))
