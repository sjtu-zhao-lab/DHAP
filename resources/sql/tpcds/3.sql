-- select  dt.d_year 
--        ,item.i_brand_id brand_id 
--        ,item.i_brand brand
--        ,sum(ss_sales_price) sum_agg
--  from  date_dim dt 
--       ,store_sales
--       ,item
--  where dt.d_date_sk = store_sales.ss_sold_date_sk
--    and store_sales.ss_item_sk = item.i_item_sk
--    and item.i_manufact_id = 816
--    and dt.d_moy=11
--  group by dt.d_year
--       ,item.i_brand
--       ,item.i_brand_id
--  order by dt.d_year
--          ,sum_agg desc
--          ,brand_id
-- ;
--  limit 100;

select  d_year 
       ,i_brand_id
       ,i_brand
       ,sum(ss_sales_price) sum_agg
 from  date_dim 
      ,store_sales
      ,item
 where d_date_sk = ss_sold_date_sk
   and ss_item_sk = i_item_sk
   and i_manufact_id = 816
   and d_moy=11
 group by d_year
      ,i_brand
      ,i_brand_id
 order by d_year
         ,sum_agg desc
         ,i_brand_id
limit 100;
;
