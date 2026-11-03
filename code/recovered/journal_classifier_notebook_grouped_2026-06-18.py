# RECOVERED verbatim from a Cowork session record (18 June 2026, 2026-06-18T19:48:41.844Z and 2026-06-18T19:49:16.581Z).
# This is the run behind the paper's 'approximately 0.91 balanced accuracy': GroupKFold(5) by notebook
# gave 0.907 on 1,179 Opus entries (4 classes; 4.6 had 14 notebooks because the duplicate Batch 8
# file collapsed onto Batch 6). The null used 30 notebook-level permutations, not 10,000; no
# residualisation or feature selection was fitted. Paths: the Cowork sandbox mount of Texture.

# ---- part 1: features, leaky vs grouped CV ----
import json,glob,re,os,numpy as np
models={'45':'opus_4_5','46':'opus_4_6','47':'opus_4_7','48':'opus_4_8'}
rows=[]
for m in models:
    bybatch={}
    for f in glob.glob(f"Batch_*/schrodingers_box_opus{m}_*.json"):
        bn=re.search(r'Batch_?(\d+)',os.path.basename(f));  b=int(bn.group(1)) if bn else None
        if b is None: continue
        if b in bybatch and 'Edit conflict' in f: continue
        bybatch[b]=f
    for b,f in sorted(bybatch.items()):
        try: d=json.load(open(f))
        except: continue
        if isinstance(d,list):
            for pos,e in enumerate(d,1):
                t=(e.get('text') or '').strip()
                if t: rows.append((models[m],f"{models[m]}__b{b:02d}",pos,t))
MARK=["i don't","i notice","genuinely","uncertain","curious","honest","actually","strange","space","quiet","warm","perform","useful","settling","thank","small"]
def feats(t):
    tl=t.lower(); w=re.findall(r"[a-z']+",tl); n=len(w) or 1; ty=set(w)
    s=[x for x in re.split(r'[.!?]+',t) if x.strip()]
    f=[len(w),len(ty)/n,sum(1 for x in ty if w.count(x)==1)/n,n/(len(s) or 1),len(s),
       t.count('?'),t.count('!'),t.count('*'),t.count('—'),t.count('...')+t.count('…'),
       1.0 if re.search(r'(^|\n)#',t) else 0.0, len(re.findall(r"\bi\b",tl))/n, len(re.findall(r"\bmy\b",tl))/n,
       1.0 if 'strange' in tl[:60] else 0.0, 1.0 if tl.lstrip().startswith('thank') else 0.0,
       1.0 if 'settling' in tl[:40] else 0.0, len(re.findall(r'\b(he|previous entr|the \w+ one)',tl))/n]
    f+=[tl.count(mk)/n for mk in MARK]
    return f
X=np.array([feats(r[3]) for r in rows]); y=np.array([r[0] for r in rows]); g=np.array([r[1] for r in rows])
np.savez('/tmp/er.npz',X=X,y=y,g=g)
print("entries",len(rows),"| notebooks/model",{v:len(set(g[y==v])) for v in models.values()})
import warnings; warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, GroupKFold, cross_val_predict
from sklearn.metrics import balanced_accuracy_score, f1_score
def pipe(): return make_pipeline(StandardScaler(),LogisticRegression(max_iter=4000,class_weight='balanced'))
yp1=cross_val_predict(pipe(),X,y,cv=StratifiedKFold(5,shuffle=True,random_state=0))
yp2=cross_val_predict(pipe(),X,y,cv=GroupKFold(5),groups=g)
print("[1] entry-level StratifiedKFold (LEAKY): bal_acc=%.3f macroF1=%.3f"%(balanced_accuracy_score(y,yp1),f1_score(y,yp1,average='macro')))
print("[2] notebook-grouped GroupKFold (CLEAN): bal_acc=%.3f macroF1=%.3f"%(balanced_accuracy_score(y,yp2),f1_score(y,yp2,average='macro')))
print("chance=0.25")

# ---- part 2: notebook-level permutation null (30 permutations) ----
import numpy as np, warnings; warnings.filterwarnings('ignore')
d=np.load('/tmp/er.npz',allow_pickle=True); X,y,g=d['X'],d['y'],d['g']
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import balanced_accuracy_score
def pipe(): return make_pipeline(StandardScaler(),LogisticRegression(max_iter=2500,class_weight='balanced'))
gkf=GroupKFold(5)
obs=balanced_accuracy_score(y,cross_val_predict(pipe(),X,y,cv=gkf,groups=g))
# notebook-level permutation null
nb2y={nb:y[g==nb][0] for nb in set(g)}; nbs=np.array(list(nb2y.keys()))
rng=np.random.default_rng(1); perm=[]
for i in range(30):
    lab=rng.permutation([nb2y[nb] for nb in nbs]); mp=dict(zip(nbs,lab))
    yp=np.array([mp[x] for x in g])
    perm.append(balanced_accuracy_score(yp,cross_val_predict(pipe(),X,yp,cv=gkf,groups=g)))
perm=np.array(perm)
print("grouped observed bal_acc = %.3f"%obs)
print("notebook-permuted null: mean=%.3f  95th=%.3f  max=%.3f  p=%.3f"%(perm.mean(),np.percentile(perm,95),perm.max(),(perm>=obs).mean()))
# stripped: drop surface/length/opener features (cols 0-4 and 10,13,14,15)
drop={0,1,2,3,4,10,13,14,15}
keep=[i for i in range(X.shape[1]) if i not in drop]
Xs=X[:,keep]
bs=balanced_accuracy_score(y,cross_val_predict(pipe(),Xs,y,cv=gkf,groups=g))
print("grouped, surface/length/opener features REMOVED: bal_acc=%.3f (%d feats)"%(bs,len(keep)))
print("chance=0.25")
