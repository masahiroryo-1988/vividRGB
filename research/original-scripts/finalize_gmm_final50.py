import time
import gmm_final50 as g
from build_gmm_final50 import build
if __name__=='__main__':
    markers=['penalty_complete.json','bayes_complete.json','stability_complete.json','implementation_tests.json']
    while not all((g.ROOT/m).exists() for m in markers):time.sleep(15)
    while any(not g.read(p)['converged'] for p in g.ROOT.glob('*/*_bayes_a*.json')):
        print('Waiting for Bayesian convergence repairs',flush=True);time.sleep(15)
    build()
