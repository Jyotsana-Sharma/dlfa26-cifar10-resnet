"""Audit, train, validate/select, promote, and export one rule-compliant model."""
import argparse, signal, shutil, subprocess, uuid
from core import *
from dataset import _open_rgb

def read_config(path):
    c=json.loads(Path(path).read_text())
    if c['precision']!='fp16' or c['augmentation']!='trivial':raise ValueError('This recipe uses FP16 and TrivialAugment')
    if c['batch_size']<1 or c['epochs']<=c['warmup_epochs'] or not 0<=c['min_lr']<=c['lr']:raise ValueError('Invalid schedule')
    if not 0<=c['label_smoothing']<1 or not 0<c['ema_decay']<1 or not 0<=c['cutout']<=32 or c['workers']<0 or not 1<=c['calibration_images']<=45000:raise ValueError('Invalid augmentation, EMA, or loader settings')
    return c

def recipe(c):return {k:v for k,v in c.items() if k not in ('name','validation_size')}
def environment():
    import torchvision,platform,PIL
    return dict(python=platform.python_version(),torch=str(torch.__version__),torchvision=str(torchvision.__version__),numpy=np.__version__,pillow=PIL.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None)

def audit(a):
    c=read_config(a.config);samples,mapping,tr,va=load_training(a.images,a.labels,c)
    hashes=[]
    for i,s in enumerate(samples):
        im=_open_rgb(s.path)
        if im.size!=(32,32):raise ValueError('Training image size mismatch')
        hashes.append(hashlib.sha256(im.tobytes()).hexdigest())
        if (i+1)%10000==0:print(f'Audit images {i+1}/50000',flush=True)
    if len(set(hashes))!=len(hashes):raise ValueError('Exact training duplicates require a grouped split; do not train this recipe yet')
    test=UnlabeledCompetitionDataset(a.sample,a.test_images,transform(),a.test_metadata)
    if [r['id'] for r in read_csv(a.test_metadata)]!=test.ids:raise ValueError('Metadata/sample order mismatch')
    train_hash=set(hashes)
    for path in test.paths:
        im=_open_rgb(path)
        if im.size!=(32,32):raise ValueError('Test size mismatch')
        if hashlib.sha256(im.tobytes()).hexdigest() in train_hash:raise ValueError('Train/test exact overlap requires inspection')
    # Smoke has no saved weights, and cannot initialize a subsequent run.
    seed_all(c['seed']);torch.set_num_threads(2);net=make_model();opt=optimizer(net,c)
    ds=CompetitionDataset(samples,mapping,transform(True,c),tr[:2]);x=torch.stack([ds[i][0] for i in range(2)]);y=torch.tensor([ds[i][1] for i in range(2)])
    before=net.stem.weight.detach().clone();v,_=sla_loss(net,x,y,c['label_smoothing']);v.backward()
    if not torch.isfinite(v) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in net.parameters()):raise ValueError('CPU loss/gradient smoke failed')
    opt.step()
    if torch.equal(before,net.stem.weight) or any(not torch.isfinite(p).all() for p in net.parameters()):raise ValueError('CPU update smoke failed')
    report=dict(status='passed',sources=source_hashes(),recipe=recipe(c),fingerprint=dataset_fingerprint(samples),
        train_count=len(tr),validation_count=len(va),index_overlap=len(set(tr)&set(va)),pixel_duplicates=0,train_test_pixel_overlap=0,
        parameters=count_params(net),cpu_smoke_loss=float(v.detach()),test_count=len(test),normalization=dict(mean=MEAN,std=STD),
        test_signature=dict(images=test_fingerprint(test),metadata=digest(a.test_metadata),sample=digest(a.sample)),environment=environment(),
        scope='Current independent PreAct pipeline/data only. Does not explain historical Wide98 score gap; no old submission replay required.')
    json_new(a.output,report);print(json.dumps(report,indent=2))

def duration(s):
    if s in ('UNLIMITED','N/A','NOT_SET'):return None
    days=0
    if '-' in s:d,s=s.split('-');days=int(d)
    p=list(map(int,s.split(':')))
    if len(p)==2:p=[0]+p
    return days*86400+p[0]*3600+p[1]*60+p[2]
def budget(seconds):
    if seconds<0:raise ValueError('seconds must be zero (no extra cap) or positive')
    seconds=float('inf') if seconds==0 else seconds
    if os.getenv('SLURM_JOB_ID') and shutil.which('squeue'):
        try:
            out=subprocess.run(['squeue','-h','-j',os.environ['SLURM_JOB_ID'],'-o','%L'],capture_output=True,text=True,check=True,timeout=10).stdout.strip()
            if not out:raise RuntimeError('No active allocation')
            for s in out.splitlines():
                v=duration(s.strip())
                if v is not None:seconds=min(seconds,v)
        except (subprocess.SubprocessError,OSError,ValueError) as exc:
            print(f'Could not read Slurm remaining time ({exc}); relying on per-epoch checkpoints and the cluster limit.',flush=True)
    if seconds<300:raise RuntimeError('Need a fresh allocation with at least five minutes remaining')
    return seconds

def train(a):
    c=read_config(a.config);gate=json.loads(Path(a.audit).read_text())
    if gate['status']!='passed' or gate['sources']!=source_hashes() or gate['recipe']!=recipe(c):raise ValueError('Run audit for this code/config first')
    if a.device!='cuda' or not torch.cuda.is_available():raise RuntimeError('Use an allocated GPU for training; audit works on CPU')
    run=Path(a.run);samples,mapping,tr,va=load_training(a.images,a.labels,c);fp=dataset_fingerprint(samples)
    if fp!=gate['fingerprint']:raise ValueError('Training data changed since audit')
    if not va:
        if not a.policy:raise ValueError('Full50K requires the frozen development policy')
        policy=json.loads(Path(a.policy).read_text())
        if policy['fingerprint']!=fp or recipe(policy['config'])!=recipe(c) or policy['validation_examples']!=5000 or policy['sources']!=source_hashes() or policy['test_signature']!=gate['test_signature']:raise ValueError('Full50K recipe differs from validated development recipe')
    if a.resume:
        ck=torch.load(run/'last.pt',map_location='cpu',weights_only=False)
        if ck['config']!=c or ck['sources']!=source_hashes() or ck['fingerprint']!=fp or ck['train_indices']!=tr or ck['validation_indices']!=va:raise ValueError('Resume config/data/code/split mismatch')
        if ck['test_signature']!=gate['test_signature'] or ck['validation_policy_sha256']!=(digest(a.policy) if a.policy else None):raise ValueError('Resume audit/policy mismatch')
    else:run.mkdir(parents=True,exist_ok=False);ck=None
    # Exclusive writer lock. SIGKILL leaves a lock; verify no process exists before removing.
    lock=run/'training.lock'
    with lock.open('x') as f:f.write(f'{os.uname().nodename} pid={os.getpid()}\n')
    try:
        train_locked(a,c,run,samples,mapping,tr,va,fp,ck,gate)
    finally:lock.unlink(missing_ok=True)

def train_locked(a,c,run,samples,mapping,tr,va,fp,ck,gate):
    seed_all(c['seed']);torch.set_num_threads(2)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    net=make_model().to(a.device);ema=copy.deepcopy(net).eval();ema.requires_grad_(False)
    opt=optimizer(net,c);scaler=torch.amp.GradScaler('cuda');start=0;step=0;history=[];best={};refs={}
    if ck:
        net.load_state_dict(ck['raw']);ema.load_state_dict(ck['ema']);opt.load_state_dict(ck['optimizer']);scaler.load_state_dict(ck['scaler']);rng_set(ck['rng'])
        start=ck['completed_epochs'];step=ck['step'];history=ck['history'];best=ck['best'];refs=ck['best_files']
        if ck['environment']!=environment():print('NOTE: runtime differs from saved environment; bitwise reproducibility is not guaranteed.',flush=True)
    stop=[False]
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:stop.__setitem__(0,True))
    deadline=time.monotonic()+budget(a.seconds);epoch_times=[r['seconds'] for r in history[-3:]]
    steps=math.ceil(len(tr)/c['batch_size'])
    def base():return dict(config=c,sources=source_hashes(),fingerprint=fp,train_indices=tr,validation_indices=va,environment=environment(),trained_from_scratch=True,test_signature=gate['test_signature'],validation_policy_sha256=digest(a.policy) if a.policy else None)
    def state(epoch):return dict(**base(),raw=cpu_state(net),ema=cpu_state(ema),optimizer=opt.state_dict(),scaler=scaler.state_dict(),rng=rng_get(),step=step,completed_epochs=epoch,history=history,best=best,best_files=refs)
    if ck is None:
        json_new(run/'config.json',c);atomic_save(run/'last.pt',state(0))
    print(json.dumps(dict(status='training',completed_epochs=start,total_epochs=c['epochs'],parameters=count_params(net),remaining_seconds=None if math.isinf(deadline) else deadline-time.monotonic())),flush=True)
    for epoch in range(start,c['epochs']):
        estimate=max(epoch_times[-3:],default=180)
        if stop[0] or deadline-time.monotonic()<estimate*1.2+120:
            print('Stopping at a saved epoch: interrupted or allocation/process budget nearly exhausted.',flush=True)
            break
        t=time.monotonic();net.train();total_loss=0.;correct=0;n=0;skips=0
        dl=loader(samples,mapping,tr,c,epoch,True)
        for batch,(x,y) in enumerate(dl):
            if stop[0] or time.monotonic()>deadline-90:
                print('Incomplete epoch discarded; resume from last committed epoch.',flush=True);return
            x=x.to(a.device,non_blocking=True);y=y.to(a.device,non_blocking=True)
            lr=learning_rate(step,steps*c['epochs'],steps*c['warmup_epochs'],c['lr'],c['min_lr'])
            for group in opt.param_groups:group['lr']=lr
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16):loss,p=sla_loss(net,x,y,c['label_smoothing'])
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss; previous completed checkpoint retained')
            old=scaler.get_scale();scaler.scale(loss).backward();scaler.step(opt);scaler.update()
            if scaler.get_scale()>=old:ema_update(ema,net,c['ema_decay'])
            else:skips+=1
            step+=1;n+=len(y);correct+=int((p.argmax(1)==y).sum());total_loss+=float(loss.detach())*len(y)
            if (batch+1)%50==0:print(f'Epoch {epoch+1}/{c["epochs"]} batch {batch+1}/{steps}',flush=True)
        metrics={}
        if va:
            for kind,m in [('raw',net),('ema',ema)]:
                print(f'Epoch {epoch+1}: validating {kind}',flush=True)
                metrics[kind]=evaluate(m,loader(samples,mapping,va,c),a.device)
        record=dict(epoch=epoch+1,train_loss=total_loss/n,train_accuracy=correct/n,validation=metrics,lr=lr,skipped_updates=skips,seconds=time.monotonic()-t)
        record['estimated_training_hours_remaining']=(c['epochs']-epoch-1)*record['seconds']/3600
        history.append(record);epoch_times.append(record['seconds'])
        for kind,m in [('raw',net),('ema',ema)]:
            if va:
                # Retain plain- and combined-selected candidates separately, no TTA tuning during training.
                for head in ('plain','combined'):
                    key=f'{kind}_{head}';acc=metrics[kind]['none/'+head]
                    if acc>best.get(key,-1):
                        name=f'{key}_e{epoch+1:03d}_{uuid.uuid4().hex[:8]}.pt'
                        atomic_save(run/name,dict(**base(),model=cpu_state(m),model_kind=kind,completed_epochs=epoch+1))
                        best[key]=acc;refs[key]=name
        atomic_save(run/'last.pt',state(epoch+1))
        atomic_json(run/'progress.json',dict(completed_epochs=epoch+1,history=history,best=best,best_files=refs))
        print(json.dumps(record),flush=True)
    if len(history)==c['epochs']:print(f'TRAINING COMPLETE: {len(history)}/{c["epochs"]} epochs saved.',flush=True)
    else:print(f'Committed {len(history)}/{c["epochs"]} epochs. Resume continues this run in the next allocation.',flush=True)

def calibration_loader(samples,mapping,tr,c,device):
    indices=np.random.default_rng(c['seed']+77).permutation(tr)[:c['calibration_images']].tolist()
    lc=c if device=='cuda' else dict(c,workers=0)
    return loader(samples,mapping,indices,lc)

def select(a):
    run=Path(a.run);last=torch.load(run/'last.pt',map_location='cpu',weights_only=False);c=last['config']
    if not last['validation_indices']:raise ValueError('Cannot select policies using full50K training data')
    if last['completed_epochs']!=c['epochs']:raise ValueError('Finish the declared development schedule before selection')
    samples,mapping,tr,va=load_training(a.images,a.labels,c);fp=dataset_fingerprint(samples)
    if fp!=last['fingerprint'] or va!=last['validation_indices'] or last['sources']!=source_hashes():raise ValueError('Validation provenance mismatch')
    all_results=[]
    for name in sorted(set(last['best_files'].values())):
        path=run/name;ck=torch.load(path,map_location='cpu',weights_only=False)
        if ck['sources']!=last['sources'] or ck['config']!=c or ck['fingerprint']!=fp or ck['validation_indices']!=va or ck['train_indices']!=tr:raise ValueError('Candidate provenance differs from run')
        m=make_model().to(a.device);m.load_state_dict(ck['model'])
        if ck['model_kind']=='ema':
            print('Calibrating EMA BN using training images only',flush=True)
            calibrate(m,calibration_loader(samples,mapping,tr,c,a.device),a.device)
        print('Evaluating '+name,flush=True)
        lc=c if a.device=='cuda' else dict(c,workers=0)
        metrics=evaluate(m,loader(samples,mapping,va,lc,batch_size=a.batch_size),a.device,('none','flip','original6'),progress=True)
        all_results.append(dict(checkpoint=str(path.resolve()),checkpoint_sha256=digest(path),model_kind=ck['model_kind'],epoch=ck['completed_epochs'],metrics=metrics))
    candidates=[(value,r,mode) for r in all_results for mode,value in r['metrics'].items()]
    # Predeclared ties: fewer views, combined head, then raw; no leaderboard information.
    def choose(options):
        value,r,mode=max(options,key=lambda t:(t[0],-{'none':1,'flip':2,'original6':6}[t[2].split('/')[0]],t[2].endswith('combined'),t[1]['model_kind']=='raw'))
        return dict(**{k:v for k,v in r.items() if k!='metrics'},tta=mode.split('/')[0],head=mode.split('/')[1],accuracy=value)
    variants={kind:choose([t for t in candidates if t[1]['model_kind']==kind and t[2]=='none/plain']) for kind in ('raw','ema')}
    variants['flip']=choose([t for t in candidates if t[2].startswith('flip/')])
    report=dict(config=c,sources=source_hashes(),fingerprint=fp,validation_examples=len(va),results=all_results,
                selected=choose(candidates),variants=variants,test_signature=last['test_signature'],
                limitation='Same 5K holdout selects epochs and inference. Development accuracy is optimistic; not an independent test or a Kaggle guarantee.')
    json_new(a.output,report);print(json.dumps(report['selected'],indent=2))

def promote(a):
    p=json.loads(Path(a.policy).read_text());c=p['config'].copy()
    if p['validation_examples']!=5000 or p['sources']!=source_hashes():raise ValueError('Need a matching development report')
    c['name']=c['name']+'_full50k';c['validation_size']=0
    json_new(a.output,c);print('Full50K scratch config written. Frozen policy: '+str(a.policy))

def predict(a):
    policy=json.loads(Path(a.policy).read_text());sel=policy['selected'] if a.variant=='selected' else policy['variants'][a.variant];out=Path(a.output)
    if out.exists():raise FileExistsError('Choose a fresh output directory')
    if policy['sources']!=source_hashes():raise ValueError('Code differs from validated policy')
    if a.checkpoint:
        ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False);checkpoint=Path(a.checkpoint)
        if ck['config']['validation_size']!=0 or ck['completed_epochs']!=ck['config']['epochs']:raise ValueError('Explicit checkpoint must be a completed full50K last.pt')
        if ck['validation_policy_sha256']!=digest(a.policy):raise ValueError('Full50K checkpoint was trained with a different frozen policy')
        weights=ck[sel['model_kind']]
    else:
        checkpoint=Path(sel['checkpoint'])
        if digest(checkpoint)!=sel['checkpoint_sha256']:raise ValueError('Selected checkpoint changed')
        ck=torch.load(checkpoint,map_location='cpu',weights_only=False);weights=ck['model']
    if recipe(ck['config'])!=recipe(policy['config']) or ck['fingerprint']!=policy['fingerprint'] or ck['sources']!=source_hashes():raise ValueError('Checkpoint provenance/recipe differs from policy')
    if ck['test_signature']!=policy['test_signature']:raise ValueError('Test provenance differs from audit')
    c=ck['config'];samples,mapping,tr,va=load_training(a.images,a.labels,c)
    if dataset_fingerprint(samples)!=ck['fingerprint']:raise ValueError('Supplied training data changed')
    net=make_model().to(a.device);net.load_state_dict(weights,strict=True);del weights,ck
    if sel['model_kind']=='ema':calibrate(net,calibration_loader(samples,mapping,tr,c,a.device),a.device)
    provenance=dict(checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint),policy_sha256=digest(a.policy),variant=a.variant)
    export_predictions(a,net,sel,c,provenance,policy['test_signature'])

def export_predictions(a,net,sel,c,provenance,test_signature):
    out=Path(a.output)
    if out.exists():raise FileExistsError('Choose a fresh output directory')
    test=UnlabeledCompetitionDataset(a.sample,a.test_images,transform(),a.test_metadata)
    signature=dict(images=test_fingerprint(test),metadata=digest(a.test_metadata),sample=digest(a.sample))
    if signature!=test_signature:raise ValueError('Test files changed since the audit')
    ids=[];labels=[];start=time.monotonic();last_log=start
    for x,bids in DataLoader(test,batch_size=a.batch_size,shuffle=False,num_workers=0):
        pred=predict_policies(net,x.to(a.device),(sel['tta'],))[sel['tta'],sel['head']]
        labels.extend(int(i) for i in pred.argmax(1).cpu().tolist());ids.extend(bids)
        elapsed=time.monotonic()-start
        if len(ids)==len(bids) or len(ids)==2000 or time.monotonic()-last_log>=10:
            print(f'Predicted {len(ids)}/2000; estimated remaining {elapsed*(2000-len(ids))/len(ids):.0f}s',flush=True)
            last_log=time.monotonic()
    out.mkdir(parents=True,exist_ok=False);write_csv(out/'submission.csv',ids,labels,test.ids)
    # Standalone final ONE-model weights, including calibrated buffers and both heads.
    atomic_save(out/'model.pt',dict(model=cpu_state(net),config=c,parameters=count_params(net),policy=sel,sources=source_hashes(),normalization=dict(mean=MEAN,std=STD),test_signature=signature,provenance=provenance))
    json_new(out/'inference.json',dict(**provenance,selected=sel,submission_sha256=digest(out/'submission.csv'),test_signature=signature,rows=2000,parameters=count_params(net),environment=environment()))
    print('DOWNLOAD THIS FILE: '+str((out/'submission.csv').resolve()),flush=True)

def replay(a):
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    if ck['sources']!=source_hashes() or ck['normalization']!=dict(mean=MEAN,std=STD):raise ValueError('Exported model/code mismatch')
    net=make_model().to(a.device);net.load_state_dict(ck['model'],strict=True)
    export_predictions(a,net,ck['policy'],ck['config'],dict(exported_model_sha256=digest(a.checkpoint)),ck['test_signature'])

def main():
    p=argparse.ArgumentParser();sp=p.add_subparsers(dest='action',required=True)
    for action in ('audit','train','select','promote','predict','replay'):
        q=sp.add_parser(action)
        if action not in ('promote','replay'):
            q.add_argument('--images',required=True);q.add_argument('--labels',required=True)
        if action in ('audit','train'):q.add_argument('--config',required=True)
        if action in ('audit','select','promote','predict','replay'):q.add_argument('--output',required=True)
        if action in ('audit','predict','replay'):
            for arg in ('test-images','test-metadata','sample'):q.add_argument('--'+arg,required=True)
        if action in ('train','select'):q.add_argument('--run',required=True)
        if action in ('promote','predict'):q.add_argument('--policy',required=True)
        if action in ('train','select','predict','replay'):q.add_argument('--device',choices=('cuda','cpu'),default='cuda')
        if action in ('select','predict','replay'):q.add_argument('--batch-size',type=int,default=64)
        if action=='train':
            q.add_argument('--audit',required=True);q.add_argument('--resume',action='store_true');q.add_argument('--seconds',type=int,default=0,help='0: no artificial cap; otherwise an extra process budget in seconds. Slurm limit still applies.');q.add_argument('--policy')
        if action=='predict':
            q.add_argument('--checkpoint');q.add_argument('--variant',choices=('selected','raw','ema','flip'),default='selected')
        if action=='replay':q.add_argument('--checkpoint',required=True)
    a=p.parse_args();torch.set_num_threads(2)
    if hasattr(a,'batch_size') and a.batch_size<1:p.error('batch-size must be positive')
    globals()[a.action](a)
if __name__=='__main__':main()
