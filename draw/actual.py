import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv(r'D:\pythonProject1\BTC-USD2019.csv', parse_dates=['Date'], index_col='Date')
# 设置字体大小
plt.rcParams.update({'font.size': 8})

fig, ax = plt.subplots(figsize=(10, 6))  # 创建图形和坐标轴对象，并设置图形大小

# 绘制折线图
ax.plot(df.index, df['Close'], color='blue', linestyle='-')

# 添加标题和标签
ax.set_xlabel('Time', fontsize=12)  # 设置x轴标签
ax.set_ylabel('Close Price (USD)', fontsize=12)  # 设置y轴标签

# 获取选中时间段的索引范围
start_date = '2020-03-01'
end_date = '2024-04-04'

# 选中时间段并将整个图形区域标红并降低透明度
ticks = ax.get_yticks()
tick_interval = ticks[1] - ticks[0]
new_max_tick = max(ticks) + tick_interval*3

# 计算新的刻度间隔（假设间隔为原始间隔）
new_tick_interval = tick_interval

# 设置新的 y 轴刻度范围和刻度间隔
ax.set_ylim(ticks[0], new_max_tick)
ax.set_yticks(np.arange(ticks[0], new_max_tick, new_tick_interval))
# 再次获取新的刻度值
ticks = ax.get_yticks()

# 重新设置 y 轴刻度范围
ax.set_ylim(ticks[0], ticks[-1])

# 计算选中时间段的纵向距离
highlight_height = ticks[-1] - ticks[0]

# 添加选中时间段的红色区域
highlight_color = 'red'
highlight_alpha = 0.3
ax.axvspan(start_date, end_date, ymin=0, ymax=(ticks[-2]/ticks[-1]), color=highlight_color, alpha=highlight_alpha)

# 显示图形
plt.tight_layout()  # 自动调整布局，防止标签重叠
plt.show()