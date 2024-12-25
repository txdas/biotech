import akshare as ak
from sklearn.svm import LinearSVC
from sklearn.preprocessing import  KBinsDiscretizer
model = LinearSVC()
model.fit
# ak.stock_zh_a_spot_em()
ak.stock_individual_fund_flow
print(ak.stock_main_fund_flow().sort_values("今日排行榜-今日排名")[["代码","名称","今日排行榜-今日排名"]][:25])