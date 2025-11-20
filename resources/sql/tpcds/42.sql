select  d_year
 	,i_category_id
 	,i_category
 	,sum(ss_ext_sales_price)
 from 	date_dim
 	,store_sales
 	,item
 where d_date_sk = ss_sold_date_sk
 	and ss_item_sk = i_item_sk
 	and i_manager_id = 1  	
 	and d_moy=12
 	and d_year=1999
 group by 	d_year
 		,i_category_id
 		,i_category
limit 100 ;
