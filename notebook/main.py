import akshare as ak
# ak.stock_zh_a_spot_em()
# ak.stock_a_indicator_lg
print(ak.stock_main_fund_flow().sort_values("今日排行榜-今日排名")[["代码","名称","今日排行榜-今日排名"]][:25])