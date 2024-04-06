import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
df = pd.read_csv(r'./DOGE 2015.9-2024.4 月度数据.csv', parse_dates=['Date'], index_col='Date')

# 设置字体大小
plt.rcParams.update({'font.size': 10})

# 将数据重新采样为每月第一个工作日的数据
monthly_open = df['Open'].resample('MS').first()

# 计算每个月相对于前一个月的收益率
df['Monthly_Return'] = monthly_open.pct_change()

# 移除可能存在的NaN值
df.dropna(subset=['Monthly_Return'], inplace=True)

# 计算平均月收益率
mean_monthly_return = df['Monthly_Return'].mean()

# 计算每月收益率的方差
df['Monthly_Variance'] = (df['Monthly_Return'] - mean_monthly_return) ** 2

# 根据年份和月份分组计算波动率
monthly_volatility = df.groupby([df.index.year, df.index.month])['Monthly_Variance'].mean() ** 0.5

# 将年份和月份转换为数值型数据
monthly_volatility.index = monthly_volatility.index.map(lambda x: str(x[0]) + '-' + str(x[1]).zfill(2))

# 提取每年的1月和7月
yearly_months = [month for month in monthly_volatility.index if month.endswith('-01') or month.endswith('-07')]


fig,ax=plt.subplots()
# 分别绘制每年的月波动率
ax.bar(range(len(monthly_volatility)), monthly_volatility.values, color='blue')

# 设置横坐标刻度和标签
ax.set_xticks(monthly_volatility.index.get_indexer(yearly_months), yearly_months,rotation=45)  # 每年的1月和7月标签


# 添加标题和标签
ax.set_xlabel('Month', fontsize=14)
ax.set_ylabel('Monthly Volatility', fontsize=14)


start_date = '2020-03'
end_date = '2024-04'

# 获取选中时间段的索引范围
start_index = monthly_volatility.index.get_loc(start_date)
end_index = monthly_volatility.index.get_loc(end_date)

# 选中时间段并将整个图形区域标红并降低透明度
highlight_color = 'red'
highlight_alpha = 0.3

# 获取纵坐标的最大值
yticks=ax.get_yticks()
print("yticks:\n",yticks)
ax.set_ylim(yticks[0],yticks[-1])

plt.axvspan(start_index - 0.5, end_index + 0.5, ymin=0, ymax=(yticks[-2]/yticks[-1]), color=highlight_color, alpha=highlight_alpha)


# 显示图形
plt.tight_layout()  # 自动调整布局，防止标签重叠
plt.show()
