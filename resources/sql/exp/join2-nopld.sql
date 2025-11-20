select p_pld0, b0_flt, b1_flt
from probe, builda, buildb
where p_key0 = b0_key
and p_key1 = b1_key