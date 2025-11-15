
import pandas as pd
import kagglehub
import os

def load_olist_cached():
    path = kagglehub.dataset_download("olistbr/brazilian-ecommerce")
    files = {
        'customers': 'olist_customers_dataset.csv',
        'orders': 'olist_orders_dataset.csv',
        'items': 'olist_order_items_dataset.csv',
        'payments': 'olist_order_payments_dataset.csv',
        'reviews': 'olist_order_reviews_dataset.csv',
    }
    dfs={}
    for k,v in files.items():
        dfs[k]=pd.read_csv(os.path.join(path,v))
    df = dfs['orders'].merge(dfs['customers'], on="customer_id", how="left")
    return df
