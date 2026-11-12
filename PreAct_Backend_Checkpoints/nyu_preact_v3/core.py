"""Single-model PreAct SLA+SD; no downloads or external weights."""
import copy, csv, hashlib, json, math, os, random, time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms as T
from models import PreActResNet, count_params
from dataset import *
MEAN=(.4914,.4822,.4465); STD=(.247,.2435,.2616)
ROOT=Path(__file__).resolve().parent
POLICIES=[(tta,head) for tta in ('none','flip','original6') for head in ('plain','aggregate','combined')]

def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()

def test_fingerprint(test):
    h=hashlib.sha256()
    for image_id,path in zip(test.ids,test.paths):
        h.update(f'{image_id}\0{path.name}\0'.encode())
        h.update(hashlib.sha256(path.read_bytes()).digest())
    return h.hexdigest()

def source_hashes():return {p.name:digest(p) for p in sorted(ROOT.glob('*.py'))}
def json_new(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(data,f,indent=2)
def atomic_save(path,data):
    path=Path(path);tmp=path.with_suffix('.tmp')
    with tmp.open('wb') as f:torch.save(data,f);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
def atomic_json(path,data):
    path=Path(path);tmp=path.with_suffix('.tmp')
    with tmp.open('w') as f:json.dump(data,f,indent=2);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
def rng_get():return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
def rng_set(r):
    random.setstate(r['python']);np.random.set_state(r['numpy']);torch.set_rng_state(r['torch'])
    if r['cuda'] is not None:torch.cuda.set_rng_state_all(r['cuda'])
def worker_seed(_):
    s=torch.initial_seed()%2**32;random.seed(s);np.random.seed(s);torch.set_num_threads(1)
class Cutout:
    def __init__(self,size):self.size=size
    def __call__(self,x):
        k=self.size;y,z=torch.randint(0,33-k,(2,)).tolist();x=x.clone();x[:,y:y+k,z:z+k]=0;return x

def transform(training=False,cfg=None):
    ops=[]
    if training:
        ops += [T.TrivialAugmentWide(),T.RandomHorizontalFlip(.5),T.RandomCrop(32,padding=4,padding_mode='reflect')]
    ops += [T.ToTensor(),T.Normalize(MEAN,STD)]
    if training:ops += [Cutout(cfg['cutout'])]
    return T.Compose(ops)
def make_model():
    net=PreActResNet((64,128,256),(4,4,3),ssl='sla')
    if count_params(net)!=4706674:raise ValueError('Unexpected architecture/parameter count')
    return net

def rotate4(x):return torch.stack([torch.rot90(x,k,(2,3)) for k in range(4)],1).flatten(0,1)
def heads(net,x):
    z=net.features(rotate4(x));j=(net.joint_head(z)*net.scale).float()
    p=(net.head(z[::4])*net.scale).float()
    a=torch.stack([j.view(len(x),4,40)[:,k,k::4] for k in range(4)]).mean(0)
    return p,a,j

def sla_loss(net,x,y,smoothing):
    p,a,j=heads(net,x);target=(y[:,None]*4+torch.arange(4,device=y.device)).flatten()
    joint=F.cross_entropy(j,target);plain=F.cross_entropy(p,y,label_smoothing=smoothing)
    sd=F.kl_div(F.log_softmax(p,1),F.softmax(a.detach(),1),reduction='batchmean')
    return joint+plain+sd,p

@torch.no_grad()
def ema_update(ema,net,decay):
    for e,p in zip(ema.parameters(),net.parameters()):e.lerp_(p,1-decay)
    for e,p in zip(ema.buffers(),net.buffers()):e.copy_(p)

def learning_rate(step,total,warm,peak,floor):
    if step<warm:return peak*(step+1)/warm
    return floor+(peak-floor)*.5*(1+math.cos(math.pi*(step-warm)/max(1,total-warm-1)))

def optimizer(net,cfg):
    return torch.optim.SGD([dict(params=[p for p in net.parameters() if p.ndim>1],weight_decay=cfg['weight_decay']),dict(params=[p for p in net.parameters() if p.ndim<=1],weight_decay=0)],lr=cfg['lr'],momentum=cfg['momentum'],nesterov=cfg['nesterov'])

def load_training(images,labels,cfg):
    samples=load_labeled_images(labels,images);mapping,_=class_mapping(samples);validate_training_distribution(samples)
    tr,va=stratified_split(samples,5000,cfg['seed'])
    if cfg['validation_size']==0:tr=list(range(len(samples)));va=[]
    elif cfg['validation_size']!=5000:raise ValueError('Only fixed 45K/5K or full50K permitted')
    return samples,mapping,tr,va

def loader(samples,mapping,indices,cfg,epoch=0,training=False,batch_size=None):
    g=torch.Generator().manual_seed(cfg['seed']+epoch*1009+(0 if training else 17))
    return DataLoader(CompetitionDataset(samples,mapping,transform(training,cfg),indices),
        batch_size=batch_size or cfg['batch_size'],shuffle=training,num_workers=cfg['workers'],
        generator=g,worker_init_fn=worker_seed,persistent_workers=False,pin_memory=torch.cuda.is_available())

def cpu_state(net):return {k:v.detach().cpu().clone() for k,v in net.state_dict().items()}

@torch.inference_mode()
def predict_policies(net,x,modes):
    if not modes or any(m not in ('none','flip','original6') for m in modes):raise ValueError('Unknown inference policy')
    net.eval();cache={}
    specs=[(0,0,False)]
    if any(m!='none' for m in modes):specs.append((0,0,True))
    if 'original6' in modes:specs.extend([(s,s,f) for s in (-1,1) for f in (False,True)])
    for y,z,flip in specs:
        view=x if y==0 and z==0 else F.pad(x,(1,)*4,mode='reflect')[:,:,1+y:33+y,1+z:33+z]
        p,a,_=heads(net,view.flip(-1) if flip else view);cache[y,z,flip]=(p,a)
    result={}
    for mode in modes:
        spec=[((0,0,False),1.)] if mode=='none' else [((0,0,f),.5) for f in (False,True)]
        if mode=='original6':spec=[((0,0,f),.25) for f in (False,True)]+[((s,s,f),.125) for s in (-1,1) for f in (False,True)]
        p=sum(cache[k][0]*w for k,w in spec).softmax(1);a=sum(cache[k][1]*w for k,w in spec).softmax(1)
        for key,value in [('plain',p),('aggregate',a),('combined',(p+a)/2)]:result[mode,key]=value
    return result

@torch.inference_mode()
def evaluate(net,dl,device,modes=('none',),progress=False):
    hits={};n=0
    for batch,(x,y) in enumerate(dl):
        preds=predict_policies(net,x.to(device),modes);n+=len(y)
        for (mode,head),p in preds.items():
            key=f'{mode}/{head}';hits[key]=hits.get(key,0)+int((p.argmax(1).cpu()==y).sum())
        if progress and (batch+1)%20==0:print(f'Validation: {n} images evaluated',flush=True)
    if not n:raise ValueError('Cannot evaluate full-data run on its training data')
    return {k:v/n for k,v in hits.items()}

@torch.no_grad()
def calibrate(net,dl,device):
    # Recompute BN from allowed TRAINING images only. Never adapt on test/holdout.
    saved={};net.eval()
    for m in net.modules():
        if isinstance(m,torch.nn.BatchNorm2d):saved[m]=m.momentum;m.reset_running_stats();m.momentum=None;m.train()
    for batch,(x,_) in enumerate(dl):
        net.features(rotate4(x.to(device)))
        if (batch+1)%20==0:print(f'EMA normalization: {batch+1}/{len(dl)} training batches',flush=True)
    for m,momentum in saved.items():m.momentum=momentum
    net.eval()

def write_csv(path,ids,labels,expected):
    if ids!=expected or len(ids)!=2000 or len(set(ids))!=2000:raise ValueError('Missing, duplicate or reordered prediction IDs')
    if len(labels)!=2000 or any(type(x)!=int or not 0<=x<=9 for x in labels):raise ValueError('Invalid labels')
    path=Path(path)
    if path.name!='submission.csv':raise ValueError('Competition requires submission.csv')
    with path.open('x',newline='') as f:
        w=csv.writer(f);w.writerow(['id','label']);w.writerows(zip(ids,labels))
    reread=read_csv(path)
    if list(reread[0])!=['id','label'] or [r['id'] for r in reread]!=expected or [int(r['label']) for r in reread]!=labels:raise ValueError('CSV roundtrip failed')
