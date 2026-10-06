import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.cluster._hdbscan.hdbscan import tree_to_labels
from threadpoolctl import threadpool_limits
import auto_clusters50 as a

with threadpool_limits(limits=2):
    raw=np.load(a.ROOT/'moin_01/whole_features.npz')['features']
    # Frozen real data exercise exact tree reuse and spatial partitions.
    samples=[raw[::4,::4,:3].reshape(-1,3),raw[::4,::4,:16].reshape(-1,16)]
    checks=0
    for x in samples:
        for ms in [3,5,10]:
            first=HDBSCAN(min_samples=ms,min_cluster_size=32,copy=False).fit(x)
            for size in [32,64,128]:
                for eps in [0.]:
                    ref=HDBSCAN(min_samples=ms,min_cluster_size=size,cluster_selection_epsilon=eps,copy=False).fit(x)
                    lab,prob=tree_to_labels(first._single_linkage_tree_.copy(),size,'eom',False,eps,None)
                    np.testing.assert_array_equal(lab,ref.labels_)
                    np.testing.assert_array_equal(prob,ref.probabilities_);checks+=1
    splits,final=a.partitions((256,256))
    for train,test in splits:
        assert len(train)==3072 and len(test)==1024 and not set(train)&set(test)
        assert len(set(train))==len(train)
    assert len(final)==8192
    a.save(a.ROOT/'implementation_tests.json',{'tree_reuse_exact_checks':checks,'partition_checks':3,'passed':True})
    print('Passed',checks,'exact tree-reuse checks and 3 partition checks')
