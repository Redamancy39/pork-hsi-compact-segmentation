"""Prepare, train and evaluate the fixed lentil protocol. Default: print plan."""
import argparse,json,os
from pathlib import Path
from lentil import common

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['prepare','train','evaluate','summarise'])
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--recover-fit-cache',action='store_true')
    args=p.parse_args()
    cfg=common.configure(args.config,execute=args.execute)
    if not args.execute:
        print(json.dumps(dict(stage=args.stage,execute=False,paths=cfg,runs=common.recipe_rows(),
              selection='six full DEV checkpoints, positive-group macro Dice at logit0',
              held_out_gate='all eight selected models locked before evaluation'),indent=2));return
    for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[name]='4'
    if args.stage=='prepare':
        common.require_data(['FIT','DEV'])
        from lentil.preparation import prepare_fit_dev
        prepare_fit_dev(recover_fit_cache=args.recover_fit_cache)
    elif args.stage=='train':
        from lentil.training import main as run
        run()
    elif args.stage=='evaluate':
        lock=common.ROOT/'locks/MODELS_LOCK.json'
        if not lock.exists() or not common.read_json(lock).get('all_eight_complete'):raise RuntimeError('All eight models must be locked before held-out data access')
        common.require_data(['DAY4_EVALUATION'])
        from lentil.evaluation import main as run
        run()
    else:
        from lentil.summary import main as run
        run()

if __name__=='__main__':main()
