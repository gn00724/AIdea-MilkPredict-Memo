#%%
from toolBox import basicTool as bt
import pandas as pd
import datetime as dt
import sys

df_pred = pd.read_csv("./submit.csv", error_bad_lines = False)
#%%
now = str(int(dt.datetime.now().timestamp()))

submit = bt.csvtoArray("./乳牛/data/submission.csv")
lo_pred = sys.argv[1]
try:
   ourput = sys.argv[2]
except:
   ourput = "./output" + now

submit
sumArray = []
titleArray = []
#%%
_tmp = 0
for _t, key in enumerate(submit):
   if _t == 0:
      titleArray = key
   else:
      sumArray.append([key[0], int(bt.AlookupSheet(lo_pred, str(key[0]), 0, [1])[0])])

   if _t/len(submit) >= _tmp + 0.01:
      print("%.2f" % (_t/len(submit) * 100) + "%")
      _tmp = _t/len(submit)
   bt.ArraytoCsvmaker("./", "output" + now, titleArray, sumArray)


     
