#!/usr/bin/env python3
"""Read-only submission comparison and PreAct V3 inference validation spot check.

Does NOT inspect private test labels, alter training or claim to identify distribution shift.
"""
import argparse
import csv
import json
import sys
from pathlib import Path


def read_csv(path, columns=('id','label')):
    with Path(path).open(newline='', encoding='utf-8-sig') as handle:
        r=csv.DictReader(handle)
        if not r.fieldnames or not set(columns).issubset(r.fieldnames):
            raise ValueError(f'{path} expected columns {columns}, found {r.fieldnames}')
        rows=list(r)
    if len(rows)!=2000:
        raise ValueError(f'{path}: expected 2000 rows; got {len(rows)}')
    ids=[row['id'] for row in rows]
    if len(set(ids))!=2000:
        raise ValueError(f'{path}: duplicated IDs')
    if not all(row['label'] in {str(x) for x in range(10)} for row in rows):
        raise ValueError(f'{path}: labels must be 0..9')
    return ids,[int(row['label']) for row in rows]


def compare(a,b,sample):
    expected,_=read_csv(sample)
    ids_a, pred_a=read_csv(a)
    ids_b, pred_b=read_csv(b)
    if expected!=ids_a or expected!=ids_b:
        raise ValueError('CSV ID order differs from the official sample_submission.csv')
    disagreements=sum(x!=y for x,y in zip(pred_a,pred_b))
    print(json.dumps({'a':str(a),'b':str(b),'rows':len(expected),
                      'prediction_disagreements':disagreements,
                      'agreement_rate':(2000-disagreements)/2000.0},indent=2))
    print('This comparison cannot reveal accuracy without hidden test labels.')


def spotcheck(project,device,batch_size,max_images):
    """Evaluate one V3-selected checkpoint twice on same REAL held-out inputs.

    Uses identical tensors with two batch sizes to catch batch-dependent inference
    and verifies the held-out data are disjoint. Not a hidden-test approximation.
    """
    import torch
    project=Path(project).resolve()
    v3=project/'nyu_preact_v3'
    study=project/'experiments/preact_sla_ta320_v3'
    selection_path=study/'selection.json'
    if not selection_path.exists():
        raise FileNotFoundError(f'V3 selection not found: {selection_path}')
    sys.path.insert(0,str(v3))
    from core import make_model, load_training, loader, predict_policies
    sel=json.loads(selection_path.read_text())
    policy=sel['selected']
    check=torch.load(policy['checkpoint'],map_location='cpu',weights_only=False)
    weights=check.get('model')
    if weights is None:
        raise ValueError('Checkpoint format unexpected: selected checkpoint lacks model state')
    if device=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but no GPU is allocated; use DEVICE=cpu to spot-check.')
    c=sel.get('config') or check.get('config')
    if c is None:raise ValueError('Missing V3 model config')
    images=Path(__import__('os').environ.get('TRAIN_IMAGES',str(Path.home()/'train_images.zip')))
    labels=Path(__import__('os').environ.get('TRAIN_LABELS',str(Path.home()/'train_labels.csv')))
    samples,mapping,tr,va=load_training(images,labels,c)
    if set(tr)&set(va):raise ValueError('Validation/training indices overlap')
    if not va:raise ValueError('This checkpoint has no validation split')
    net=make_model().to(device)
    net.load_state_dict(weights,strict=True)
    net.eval()
    if policy['model_kind']=='ema':
        raise RuntimeError('EMA requires training-only BN recalibration; use V3 selection to evaluate EMA rather than this raw spot check')
    mode=policy['tta'];head=policy['head']
    if not 0<max_images<=len(va):raise ValueError('max_images must be within validation split')
    indices=va[:max_images]
    dl=loader(samples,mapping,indices,c,training=False,batch_size=batch_size)
    labels_a=[];pred_a=[]
    with torch.inference_mode():
        for x,y in dl:
            scores=predict_policies(net,x.to(device),(mode,))[(mode,head)]
            pred_a.extend(scores.argmax(1).cpu().tolist());labels_a.extend(y.tolist())
    # One example at a time; same model, same data, same selected policy.
    dl2=loader(samples,mapping,indices,c,training=False,batch_size=1)
    pred_b=[]
    with torch.inference_mode():
        for x,y in dl2:
            pred_b.extend(predict_policies(net,x.to(device),(mode,))[(mode,head)].argmax(1).cpu().tolist())
    hits=sum(x==y for x,y in zip(pred_a,labels_a))
    differences=sum(x!=y for x,y in zip(pred_a,pred_b))
    out={'checkpoint':str(policy['checkpoint']), 'model_kind':policy['model_kind'],
         'policy':mode+'/'+head, 'validation_images_checked':len(indices),
         'accuracy_subset':hits/len(indices),'batch_size_comparison':batch_size,
         'prediction_differences_between_batches':differences,
         'limitation':'Batch-size and heldout accuracy sanity check only; does not reproduce the Kaggle submission or measure private test accuracy.'}
    print(json.dumps(out,indent=2))
    if differences:raise RuntimeError('Batch-dependent predictions found; investigate before retraining')


def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='action',required=True)
    q=sub.add_parser('compare');q.add_argument('--a',required=True);q.add_argument('--b',required=True);q.add_argument('--sample',required=True)
    q=sub.add_parser('v3');q.add_argument('--project',required=True);q.add_argument('--device',default='cpu',choices=['cpu','cuda']);q.add_argument('--batch-size',type=int,default=32);q.add_argument('--max-images',type=int,default=256)
    args=p.parse_args()
    if args.action=='compare':compare(args.a,args.b,args.sample)
    else:spotcheck(args.project,args.device,args.batch_size,args.max_images)

if __name__=='__main__':main()
