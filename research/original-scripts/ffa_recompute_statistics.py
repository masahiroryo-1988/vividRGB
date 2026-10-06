"""Portable saved-result verification. Requires Python and NumPy, no GPU."""
import argparse,itertools,json,hashlib,csv
from pathlib import Path
import numpy as np

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,default=Path(__file__).resolve().parent.parent/'data');a=ap.parse_args()
    p=a.data/'ecology-collection75_results.json';raw=json.loads(p.read_text(encoding='utf-8'));e=json.loads((a.data/'evidence.json').read_text(encoding='utf-8'))
    assert hashlib.sha256(p.read_bytes()).hexdigest()==e['source_sha256']
    rows=raw['images'];assert len(rows)==50
    methods=['whole','sam3','s05','s10'];domains=['moin','fungal_network','neon','coralscapes','pmid']
    def metric(r,m):return r['methods'][m]['metrics']['quality'] if m=='sam3' else r['methods'][m]['k']['6']
    def f1_check(z):
        p,r=z['precision'],z['recall'];f=2*p*r/(p+r) if p+r else 0.
        assert abs(f-z['f1'])<1e-12
    signs=np.array(list(itertools.product([-1,1],repeat=10)));checks=[];contrasts=[]
    for ds in domains:
        rr=[x for x in rows if x['dataset']==ds];assert len(rr)==10
        for m in methods:
            y=np.array([metric(r,m)['rgb_edge']['f1'] for r in rr])
            for r in rr:f1_check(metric(r,m)['rgb_edge'])
            expected=e['domains'][ds]['rgb'][m]
            np.testing.assert_allclose([y.mean(),y.std(ddof=1)],[expected['mean'],expected['sd']],atol=1e-12,rtol=0)
            checks.append({'dataset':ds,'method':m,'mean':float(y.mean()),'sd':float(y.std(ddof=1))})
        for m in ['s05','s10']:
            for b in ['whole','sam3']:
                dif=np.array([metric(r,m)['rgb_edge']['f1']-metric(r,b)['rgb_edge']['f1'] for r in rr]);null=np.abs(signs@dif/10);p=float(np.mean(null>=abs(dif.mean())-1e-14))
                contrasts.append({'dataset':ds,'focus':m,'baseline':b,'mean_delta':float(dif.mean()),'raw_p':p})
    order=sorted(range(20),key=lambda i:contrasts[i]['raw_p']);last=0
    for rank,i in enumerate(order):last=max(last,min(1.,contrasts[i]['raw_p']*(20-rank)));contrasts[i]['holm_p']=last
    for z,expected in zip(contrasts,e['contrasts']):
        np.testing.assert_allclose([z['mean_delta'],z['raw_p'],z['holm_p']],[expected['mean_delta'],expected['exact_two_sided_p'],expected['holm_p']],atol=1e-12,rtol=0)
    for m in methods:
        rr=[r for r in rows if r['dataset']=='coralscapes'];v=[]
        for r in rr:
            z=metric(r,m)['reference']['boundary'];f1_check(z);v.append(z['f1'])
        np.testing.assert_allclose(np.mean(v),e['domains']['coralscapes']['annotated'][m]['mean'],atol=1e-12,rtol=0)
    # Verify Figure 7 coordinates and arithmetic means against the frozen source.
    cost_checks=None
    cost_path=a.data/'cost_tradeoff_summary.json'
    if cost_path.exists():
        cost=json.loads(cost_path.read_text(encoding='utf-8'))
        assert cost['source_sha256']==e['source_sha256']
        with (a.data/'cost_tradeoff_points.csv').open(newline='',encoding='utf-8') as f:
            points=list(csv.DictReader(f))
        assert len(points)==200
        source={r['id']:r for r in rows}
        seen=set();missing=0;mean_checks=0
        keys=['rgb_edge_f1','extraction_seconds','gpu_board_joules','gpu_board_efficiency_inverse_joules']
        for point in points:
            r=source[point['image']];m=point['method'];ex=r['methods'][m]['extraction']
            assert point['dataset']==r['dataset'] and (r['id'],m) not in seen
            seen.add((r['id'],m))
            expected=[metric(r,m)['rgb_edge']['f1'],ex['seconds']]
            if m=='whole':
                assert point['gpu_board_joules']==point['gpu_board_efficiency_inverse_joules']==''
                missing+=1
            else:
                expected += [ex['gpu_board_joules'],1/ex['gpu_board_joules']]
            np.testing.assert_allclose([float(point[k]) for k in keys[:len(expected)]],expected,atol=1e-12,rtol=0)
            assert .08<float(point['extraction_seconds'])<650
            if m!='whole':assert 3.8e-5<float(point['gpu_board_efficiency_inverse_joules'])<5.6e-3
            assert 0<=float(point['rgb_edge_f1'])<.88
        assert missing==50
        for ds in [None]+domains:
            for m in methods:
                selected=[p for p in points if p['method']==m and (ds is None or p['dataset']==ds)]
                saved=(cost['overall'] if ds is None else cost['domains'][ds])[m]
                for key in keys:
                    values=[float(p[key]) for p in selected if p[key]!='']
                    if not values:assert saved[key] is None;continue
                    assert saved[key]['n']==len(values)
                    np.testing.assert_allclose([np.mean(values),np.std(values,ddof=1)],
                                               [saved[key]['mean'],saved[key]['sd']],atol=1e-12,rtol=0)
                    mean_checks+=1
        cost_checks={'image_method_coordinates':200,'efficiency_coordinates':150,
                     'missing_dinov3_energy_fields':50,'mean_sd_checks':mean_checks,
                     'all_observations_within_plot_limits':True}
    report={'complete':True,'images':50,'rgb_f1_recalculations':200,'annotated_f1_recalculations':40,'mean_sd_checks':20,'paired_tests_and_holm_checks':20,'figure7':cost_checks,'source_sha256':e['source_sha256'],'summaries':checks,'contrasts':contrasts}
    (a.data/'independent_numeric_validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ['summaries','contrasts']}))
if __name__=='__main__':main()
