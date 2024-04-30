import pandas as pd
import matplotlib.pyplot as plt
import numpy as np  # 需要导入numpy库

# 读取数据文件
data1 = pd.read_csv('GM_EWPT_Scan666.dat', delimiter=' ', header=None, names=['col1', 'col2', 'col3', 'col4', 'col5', 'col6', 'col7', 'col8', 'col9'])

#合并两个数据
#merged_data=pd.concat([data1])

# 筛选出第六列数据不为-1的行
filtered_data = data1[data1['col6'] != -1]

# 选择第一列和第二列中较小的一个作为横轴，第五列数据作为纵轴
x_values = filtered_data[['col1', 'col2']].min(axis=1)
y_values = filtered_data['col5']

# 绘制散点图
plt.scatter(x_values, y_values, label='Data Points')

# 添加 x/y=10 的线
x_line = np.linspace(min(x_values), max(x_values), 400)  # 生成x轴的数据点
y_line10 = x_line / 10
plt.plot(x_line, y_line10, 'r--', label='x/y = 10')  # 'r--' 表示红色虚线

# 添加 x/y=20 的线
y_line20 = x_line / 20
plt.plot(x_line, y_line20, 'b--', label='x/y = 20')  # 'b--' 表示蓝色虚线

# 添加 x/y=28 的线
y_line28 = x_line / 28
plt.plot(x_line, y_line28, 'g--', label='x/y = 28')  # 'g--' 表示绿色虚线

# 设置标签和标题
plt.xlabel('X Axis')
plt.ylabel('Y Axis')
plt.title('Scatter Plot with Lines x/y = 10 and x/y = 20 and x/y = 28')
plt.legend()  # 显示图例

# 显示图形
plt.show()
