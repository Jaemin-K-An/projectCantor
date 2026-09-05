from __future__ import annotations

import hashlib
import json
import pathlib
import sys

import numpy as np

ROOT=pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'llm/src'))
sys.path.insert(0,str(ROOT))
CONFIG=ROOT/'configs/v3_6_0'
RESULTS=ROOT/'results/v3_6_0'
FIGURES=ROOT/'figures/v3_6_0'
BASE='9e525e886fcf68bff61a8735b2e56c1d8d145908'
W_R=1.3727408729684236
RHOS=(.25,.28,.30,1/3,.36,.40,.44)
SENSOR_SHA='f16942ce8c6f89d2eaee2679da4778156450cd44fe1b9ac3529f3434f402f1fe'
ACTUATOR_SHA='c22957e2fe05e9fa3bc158853dbb5c88965b62a98c2aefd63f11fa73d480172a'


def read(path): return json.loads(pathlib.Path(path).read_text())
def sha(path): return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
def write(path,value):
    def clean(x):
        if isinstance(x,dict): return {str(k):clean(v) for k,v in x.items()}
        if isinstance(x,(list,tuple)): return [clean(v) for v in x]
        if isinstance(x,np.ndarray): return clean(x.tolist())
        if isinstance(x,np.generic): return clean(x.item())
        if isinstance(x,float) and not np.isfinite(x): return None
        return x
    p=pathlib.Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(clean(value),indent=2,allow_nan=False)+'\n')


def assets():
    from cantor_guard_v340.sensor_distance import SensorHyperplane
    w=ROOT/'results/v3_4_0/cache/sensor_w.npy'
    v=ROOT/'results/v3_3_5a/cache/v_p0.npy'
    if sha(w)!=SENSOR_SHA or sha(v)!=ACTUATOR_SHA:
        raise RuntimeError('frozen asset hash mismatch')
    sensor=SensorHyperplane(np.load(w),read(ROOT/'results/v3_4_0/tables/sensor_confirm.json')['b'])
    actuator=np.load(v).astype(float).reshape(-1)
    actuator/=np.linalg.norm(actuator)
    return sensor,actuator


def freeze_check():
    f=read(CONFIG/'PRE_ANALYSIS_FREEZE.json')
    if f['status']!='FROZEN': raise RuntimeError('research protocol is not frozen')
    for path,expected in f['hashes'].items():
        if sha(ROOT/path)!=expected: raise RuntimeError(f'post-freeze mutation: {path}')
    assets()
    return read(CONFIG/'protocol.json')


def save_table(name,frame):
    p=RESULTS/'raw'/name;p.parent.mkdir(parents=True,exist_ok=True)
    frame.to_csv(p,index=False)
