from optimal_subset_CVV_Linf import OptimalSubset_CVV_Linf
from optimal_subset_CVV_L1 import OptimalSubset_CVV_L1
from optimal_subset_AVV_Linf import OptimalSubset_AVV_Linf
from optimal_subset_AVV_L1 import OptimalSubset_AVV_L1
from optimal_subset_Classic import OptimalSubset_Classic

ALGO_MAP = {
    ('CVV', 'Linf'): OptimalSubset_CVV_Linf,
    ('CVV', 'L1'): OptimalSubset_CVV_L1,
    ('AVV', 'Linf'): OptimalSubset_AVV_Linf,
    ('AVV', 'L1'): OptimalSubset_AVV_L1,
    (None, None): OptimalSubset_Classic,
}
