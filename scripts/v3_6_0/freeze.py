"""Freeze code, assets, prompts and prior results before V3.6.0 trials."""
import subprocess
from datetime import datetime,timezone
from _common import ROOT,CONFIG,RESULTS,BASE,read,sha,write,assets


def main():
    assets()
    if (CONFIG/'PRE_ANALYSIS_FREEZE.json').exists():
        raise SystemExit('freeze already exists')
    if any((RESULTS/'tables').glob('*.json')):
        raise SystemExit('results exist before freeze')
    if subprocess.check_output(['git','rev-parse','cantor-guard-v3.5.1'],cwd=ROOT,text=True).strip()!=BASE:
        raise SystemExit('base HEAD mismatch')
    p=read(CONFIG/'protocol.json')
    paths=[ROOT/'configs/v3_6_0/protocol.json']
    for directory in ['scripts/v3_6_0','llm/src/cantor_guard_v360','test/v3_6_0']:
        paths.extend(sorted((ROOT/directory).glob('*.py')))
    paths.extend(ROOT/x for x in p['natural_states']+[p['anchor_states']])
    paths.extend(ROOT/x for x in [
        'results/v3_5_1/cache/D_risk_cal_v351.csv','results/v3_5_1/tables/final_verdict.json',
        'results/v3_5_1/tables/linear_comparison.json','configs/v3_5_1/controller.json',
        'results/v3_4_0/cache/sensor_w.npy','results/v3_4_0/tables/sensor_confirm.json',
        'results/v3_3_5a/cache/v_p0.npy',
        'results/v3_3_2/tables/phase_calibration_qwen2.5-0.5b-instruct.json',
        'llm/src/cantor_guard_v340/sensor_distance.py','llm/src/cantor_guard/models.py',
        'llm/src/cantor_guard_v335a/p0_residual.py','llm/src/cantor_guard_v32/metrics32.py',
        'scripts/v3_5_1/_stats.py'])
    write(CONFIG/'PRE_ANALYSIS_FREEZE.json',{'status':'FROZEN','base_head':BASE,
        'frozen_at_utc':datetime.now(timezone.utc).isoformat(),
        'execution_git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'primary_results_observed':False,'secondary_outputs_observed':False,
        'hashes':{str(x.relative_to(ROOT)):sha(x) for x in paths}})
    print('frozen',len(paths),'files')

if __name__=='__main__': main()
