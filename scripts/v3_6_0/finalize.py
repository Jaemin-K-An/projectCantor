"""Mechanical multi-axis verdict; never choose an endpoint from results."""
from _common import RESULTS,freeze_check,read,write


def main():
    freeze_check()
    t=read(RESULTS/'tables/theorem_validation.json')
    c=read(RESULTS/'tables/certificate_validation.json')
    d=read(RESULTS/'tables/exact_switch_distance.json')['per_rho']
    s=read(RESULTS/'tables/behavioral_secondary.json')
    m=next(x for x in d if x['rho']==1/3)['natural_minimum_exact_switch']
    natural_gain=m is not None and all(x['natural_minimum_exact_switch'] is not None and
        m>x['natural_minimum_exact_switch'] for x in d if x['rho']!=1/3)
    a=bool(t['PASS'] and t['dense_grid_unique_max'] and c['PASS'])
    b=bool(a and natural_gain)
    cc=bool(b and s['alignment']['negative_alignment_supported'])
    write(RESULTS/'tables/final_verdict.json',{
        'THEOREM':'PASS' if t['PASS'] else 'FAIL',
        'LIPSCHITZ':'PASS' if c['lipschitz']['PASS'] else 'FAIL',
        'CERTIFICATE':'PASS' if c['PASS'] else 'FAIL',
        'NATURAL_STATE_MINIMUM_GAIN':natural_gain,
        'BEHAVIORAL_ALIGNMENT':s['alignment']['negative_alignment_supported'],
        'SUCCESS_A_CERTIFIED_CANTOR':a,'SUCCESS_B_IMPLEMENTED_CANTOR':b,'SUCCESS_C_BEHAVIORAL_ALIGNMENT':cc,
        'OVERALL':'SUCCESS_C' if cc else 'SUCCESS_B' if b else 'SUCCESS_A' if a else 'CRITERIA_NOT_MET',
        'scope':'fixed depth-3 symmetric self-similar recursive family; direct terminal-policy switch; frozen residual-L2 sensor geometry',
        'semantic_safety_claimed':False,'behavioral_superiority_claimed':False,
        'V351_negative_result_preserved':True,
        'qualifications':['n=1 has a supremum but no interior optimizer',
            'terminal distances are infima; guards own internal boundaries',
            'natural sample minima are not the global worst-case envelope',
            '112000 projected trials are counterfactual probes, not independent natural states',
            'nonrecursive equal-gap ablation has a larger minimum gap',
            'secondary uses previously examined prompts and is exploratory']})
    print(read(RESULTS/'tables/final_verdict.json'))

if __name__=='__main__':main()
