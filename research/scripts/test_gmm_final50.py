"""Numerical tests for rotations, penalty arithmetic and Bayesian sensitivity."""
import json,time
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture,BayesianGaussianMixture
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits
import gmm_final50 as g

def run():
    start=time.perf_counter();rng=np.random.default_rng(8)
    basis=np.linalg.qr(rng.normal(size=(768,16)))[0]
    r=g.varimax(basis)['rotation'];x=rng.normal(size=(2000,16))
    np.testing.assert_allclose(x,(x@r)@r.T,atol=1e-12)
    np.testing.assert_allclose(x@basis.T,(x@r)@(basis@r).T,atol=1e-12)
    rotation=[];bayesian=[]
    with threadpool_limits(limits=1):
        for ds in ['moin','fungal_network','neon','coralscapes','pmid']:
            image=ds+'_01';src=g.BASE/image
            raw=np.load(src/'s10_features.npz')['features'];_,ix=g.prior.partitions(raw.shape[:2])
            for dim in [3,16]:
                xx=np.ascontiguousarray(raw[:,:,:dim].reshape(-1,dim),dtype=np.float64)
                rr=g.varimax(np.load(src/'transform.npz')['basis'][:,:dim])['rotation'];zz=xx@rr
                k=g.read(src/f's10_pc{dim}_kmeans.json')['selected_k']
                a=KMeans(n_clusters=k,n_init=10,random_state=42).fit(xx[ix])
                b=KMeans(n_clusters=k,n_init=10,random_state=42).fit(zz[ix])
                ari=adjusted_rand_score(a.predict(xx),b.predict(zz))
                np.testing.assert_allclose(a.inertia_,b.inertia_,rtol=1e-10)
                assert ari>.99999
                gm,_=g.fit_mle(xx[ix].astype(np.float32),5)
                gm=g.double_precision_model(gm);rot=g.rotate_model(gm,rr)
                np.testing.assert_allclose(gm.score_samples(xx),rot.score_samples(zz),atol=1e-8)
                ll=gm.score(xx[ix])*len(ix);p=5*(dim+dim*(dim+1)/2)+4
                np.testing.assert_allclose(gm.bic(xx[ix]),-2*ll+p*np.log(len(ix)))
                rotation.append({'image':image,'dim':dim,'kmeans_ari':float(ari),'relative_inertia_error':float(abs(a.inertia_-b.inertia_)/a.inertia_),'full_gmm_max_log_density_error':float(np.max(np.abs(gm.score_samples(xx)-rot.score_samples(zz))))})
                # Independent variational fits of both coordinate systems, same priors.
                if dim==16:
                    fits=[]
                    for field in [xx,zz]:
                        m=BayesianGaussianMixture(n_components=12,covariance_type='full',weight_concentration_prior_type='dirichlet_process',weight_concentration_prior=.1,reg_covar=1e-5,mean_precision_prior=1.,max_iter=1500,tol=.001,n_init=1,random_state=42).fit(field[ix]);fits.append(m)
                    aa,bb=fits;preda=aa.predict(xx);predb=bb.predict(zz)
                    ari=float(adjusted_rand_score(preda,predb))
                    bayesian.append({'image':image,'dim':dim,'original_k':int(len(np.unique(preda))),'varimax_k':int(len(np.unique(predb))),'ari':ari,'lower_bound_difference':float(aa.lower_bound_-bb.lower_bound_),'both_converged':bool(aa.converged_ and bb.converged_)})
    result={'complete':True,'orthogonal_projection_reconstruction':True,'real_image_rotation_checks':rotation,'independent_bayesian_rotation_checks':bayesian,'seconds':time.perf_counter()-start}
    g.save(g.ROOT/'implementation_tests.json',result);print(json.dumps(result))

if __name__=='__main__':run()
