import pandas as pd
t2c = pd.read_pickle(r"c:\Users\serveradmin\Desktop\resume-processing-system\data\title_to_cluster.pkl")
print(type(t2c))
if hasattr(t2c, 'columns'):
    print(t2c.columns.tolist())
    print(t2c.head(3))
elif hasattr(t2c, 'index'):
    print("Series, first 3:")
    print(t2c.head(3))
else:
    print("Dict-like, first 3:")
    items = list(t2c.items())[:3]
    print(items)
