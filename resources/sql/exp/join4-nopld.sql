select p_pld0, b0_flt, b1_flt, b2_flt, b3_flt
from probe, builda, buildb, buildc, buildd
where p_key0 = b0_key
and p_key1 = b1_key
and p_key2 = b2_key
and p_key3 = b3_key