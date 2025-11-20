select p_pld0, b0_flt, b0_pld0, b0_pld1, b1_flt, b1_pld0, b1_pld1, b2_flt, b2_pld0, b2_pld1
from probe, builda, buildb, buildc
where p_key0 = b0_key
and p_key1 = b1_key
and p_key2 = b2_key