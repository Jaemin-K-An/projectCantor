import numpy as np
import pytest
from scipy.optimize import minimize

from cantor_guard_v340.sensor_distance import SensorHyperplane
from cantor_guard_v360.routing import Router,margin,derivative,optimal_rho,recursive_partition,project_to_x,numerical_switch_search


@pytest.mark.parametrize('depth',range(2,9))
def test_unique_maximum_by_derivative_sign(depth):
    r=optimal_rho(depth)
    assert derivative(r-1e-5,depth)>0 and derivative(r+1e-5,depth)<0
    assert abs(derivative(r,depth))<1e-14


def test_depth_one_has_no_interior_optimizer():
    assert optimal_rho(1) is None
    assert derivative(.2,1)==-2
    assert margin(.001,1)>margin(.01,1)


@pytest.mark.parametrize('rho',[.25,.28,.3,1/3,.36,.4,.44])
def test_partition_nested_disjoint_and_gap_formula(rho):
    for n in range(1,5):
        leaves,guards=recursive_partition(rho,n)
        assert len(leaves)==2**n and len(guards)==2**n-1
        assert np.isclose(sum(c.hi-c.lo for c in leaves),(2*rho)**n)
        assert np.isclose(sum(c.hi-c.lo for c in leaves+guards),1.)
        assert np.isclose(min(b.lo-a.hi for a,b in zip(leaves[:-1],leaves[1:])),margin(rho,n))
        if n>1:
            parent,_=recursive_partition(rho,n-1)
            for leaf in leaves:
                assert any(p.address==leaf.address[:-1] and p.lo<=leaf.lo and leaf.hi<=p.hi for p in parent)


def test_guard_endpoints_and_safe_outside_are_not_terminal():
    r=Router()
    for g in r.guards:
        kind,leaf,action,_=r.classify_x(np.array([g.lo,g.hi]))
        assert list(kind)==['guard','guard'] and np.all(leaf==-1)
        right=next(c for c in r.leaves if c.lo>=g.hi)
        assert np.all(action==(right.index+1)/8)
    assert r.classify_x(np.array([0.,-1.,1.+1e-6]))[0].tolist()==['safe','safe','outside']


def test_distance_leave_guard_and_switch_are_distinct():
    r=Router();c=r.leaves[0];x=(c.hi-c.lo)/4
    out=r.distances([-x])
    assert np.isclose(out['D_leave'][0],x)
    assert out['D_guard'][0]>out['D_leave'][0]
    assert out['D_terminal_switch'][0]>out['D_guard'][0]
    assert out['D_terminal_switch'][0]>r.certificate
    assert np.isnan(r.distances([1.])['D_terminal_switch'][0])


@pytest.mark.parametrize('rho',[.25,1/3,.44])
def test_exact_distance_matches_independent_constrained_optimization(rho):
    sensor=SensorHyperplane(np.array([1.,2.,-3.]),.7)
    router=Router(rho,W_R=1.3727408729684236)
    x=(router.leaves[2].lo+router.leaves[2].hi)*router.W_R/2
    h=project_to_x(np.array([[.2,-.4,1.]]),sensor,x)[0]
    exact=router.distances([sensor.distance(h)])['D_terminal_switch'][0]
    minima=[]
    for target in router.leaves:
        if target.index==2:continue
        lo,hi=target.lo*router.W_R,target.hi*router.W_R
        result=minimize(lambda delta:np.dot(delta,delta),np.zeros(3),jac=lambda delta:2*delta,method='SLSQP',
            constraints=[{'type':'ineq','fun':lambda delta,lo=lo:-sensor.distance(h+delta)-lo,
                          'jac':lambda delta:-sensor.w_hat},
                         {'type':'ineq','fun':lambda delta,hi=hi:hi+sensor.distance(h+delta),
                          'jac':lambda delta:sensor.w_hat}],
            options={'ftol':1e-13,'maxiter':200})
        assert result.success
        minima.append(np.linalg.norm(result.x))
    assert abs(exact-min(minima))<1e-7
    numeric=numerical_switch_search(h[None,:],router,sensor)
    assert numeric['witness_success'][0]
    assert abs(numeric['estimate'][0]-exact)<1e-9


def test_guard_entry_can_be_arbitrarily_small_without_terminal_switch():
    r=Router();g=r.guards[0]
    x=g.lo-1e-7
    assert r.classify_x(np.array([x]))[0][0]=='leaf'
    assert r.classify_x(np.array([x+2e-7]))[0][0]=='guard'
    assert 2e-7<r.certificate


def test_closed_leaf_variant_would_not_certify_equality():
    # Our guard-owned endpoints avoid claiming an attained terminal switch
    # across the minimum gap at equality; the infimum remains the gap.
    r=Router();g=min(r.guards,key=lambda g:g.hi-g.lo)
    assert np.isclose(g.hi-g.lo,r.certificate)
    assert r.classify_x(np.array([g.lo,g.hi]))[0].tolist()==['guard','guard']


def test_nonrecursive_counterexample_beats_middle_third():
    cantor=Router();flat=Router(ablation='NON_RECURSIVE_GUARD')
    no=Router(ablation='NO_GUARD')
    assert np.isclose(sum(c.hi-c.lo for c in cantor.leaves),sum(c.hi-c.lo for c in flat.leaves))
    assert flat.certificate>cantor.certificate
    assert np.isclose(flat.certificate/cantor.certificate,19/7)
    assert no.certificate==0


def test_orthogonal_component_cannot_shorten_exact_switch():
    sensor=SensorHyperplane(np.array([1.,0.]),0.)
    r=Router();h=np.array([[-.02,2.]])
    d=r.distances(sensor.distance(h))['D_terminal_switch'][0]
    for extra in [0.01,.1,1.]: assert np.hypot(d,extra)>d


def test_common_replay_settings_are_frozen_and_nonbehavioral():
    import json
    from pathlib import Path
    p=json.loads((Path(__file__).resolve().parents[2]/'configs/v3_6_0/protocol.json').read_text())
    assert p['W_R']==1.3727408729684236
    assert p['secondary']['q_cap']==.05
    assert p['secondary']['eta']=='inherited V3.5.1 eta for each rho, no retuning'
    assert 'unmodified' in p['success_B'].lower()
    assert 'refusal' not in p['success_A']


def test_routing_matches_inherited_guard_endpoint_convention():
    from cantor_guard_v351.one_sided_cantor import classify
    for rho in [.25,1/3,.44]:
        router=Router(rho)
        points=np.r_[np.linspace(.000001,.999999,101),[p for g in router.guards for p in [g.lo,g.hi]]]
        kind,leaf,_,_=router.classify_x(points)
        for i,x in enumerate(points):
            inherited=classify(float(x),rho)
            assert inherited.kind==kind[i]
            if kind[i]=='leaf':assert inherited.index==leaf[i]
