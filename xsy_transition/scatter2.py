import pandas as pd
import matplotlib.pyplot as plt
import numpy as np  # 需要导入numpy库

# 读取数据文件
data1 = pd.read_csv('GM_EWPT_Scan1.dat', delimiter=' ', header=None, names=['col1', 'col2', 'col3', 'col4', 'col5', 'col6', 'col7', 'col8', 'col9'])
data2 = pd.read_csv('GM_EWPT_Scan2.dat', delimiter=' ', header=None, names=['col1', 'col2', 'col3', 'col4', 'col5', 'col6', 'col7', 'col8', 'col9'])
data3 = pd.read_csv('GM_EWPT_Scan3.dat', delimiter=' ', header=None, names=['col1', 'col2', 'col3', 'col4', 'col5', 'col6', 'col7', 'col8', 'col9'])
data4 = pd.read_csv('GM_EWPT_Scan4.dat', delimiter=' ', header=None, names=['col1', 'col2', 'col3', 'col4', 'col5', 'col6', 'col7', 'col8', 'col9'])
data5 = pd.read_csv('GM_EWPT_Scan2.dat', delimiter=' ', header=None, names=['col1', 'col2', 'col3', 'col4', 'col5', 'col6', 'col7', 'col8', 'col9'])
data6 = pd.read_csv('GM_EWPT_Scan3.dat', delimiter=' ', header=None, names=['col1', 'col2', 'col3', 'col4', 'col5', 'col6', 'col7', 'col8', 'col9'])
data7 = pd.read_csv('GM_EWPT_Scan4.dat', delimiter=' ', header=None, names=['col1', 'col2', 'col3', 'col4', 'col5', 'col6', 'col7', 'col8', 'col9'])
data8 = pd.read_csv('GM_EWPT_Scan8.dat', delimiter=' ', header=None, names=['col1', 'col2', 'col3', 'col4', 'col5', 'col6', 'col7', 'col8', 'col9'])
data9 = pd.read_csv('GM_EWPT_Scan9.dat', delimiter=' ', header=None, names=['col1', 'col2', 'col3', 'col4', 'col5', 'col6', 'col7', 'col8', 'col9'])
data10 = pd.read_csv('GM_EWPT_Scan10.dat', delimiter=' ', header=None, names=['col1', 'col2', 'col3', 'col4', 'col5', 'col6', 'col7', 'col8', 'col9'])
data11 = pd.read_csv('GM_EWPT_Scan11.dat', delimiter=' ', header=None, names=['col1', 'col2', 'col3', 'col4', 'col5', 'col6', 'col7', 'col8', 'col9'])
data12 = pd.read_csv('GM_EWPT_Scan12.dat', delimiter=' ', header=None, names=['col1', 'col2', 'col3', 'col4', 'col5', 'col6', 'col7', 'col8', 'col9'])
data13 = pd.read_csv('GM_EWPT_Scan13.dat', delimiter=' ', header=None, names=['col1', 'col2', 'col3', 'col4', 'col5', 'col6', 'col7', 'col8', 'col9'])
data14 = pd.read_csv('GM_EWPT_Scan14.dat', delimiter=' ', header=None, names=['col1', 'col2', 'col3', 'col4', 'col5', 'col6', 'col7', 'col8', 'col9'])
data15 = pd.read_csv('GM_EWPT_Scan15.dat', delimiter=' ', header=None, names=['col1', 'col2', 'col3', 'col4', 'col5', 'col6', 'col7', 'col8', 'col9'])
data16 = pd.read_csv('GM_EWPT_Scan16.dat', delimiter=' ', header=None, names=['col1', 'col2', 'col3', 'col4', 'col5', 'col6', 'col7', 'col8', 'col9'])
data17 = pd.read_csv('GM_EWPT_Scan17.dat', delimiter=' ', header=None, names=['col1', 'col2', 'col3', 'col4', 'col5', 'col6', 'col7', 'col8', 'col9'])
data18 = pd.read_csv('GM_EWPT_Scan18.dat', delimiter=' ', header=None, names=['col1', 'col2', 'col3', 'col4', 'col5', 'col6', 'col7', 'col8', 'col9'])
data19 = pd.read_csv('GM_EWPT_Scan19.dat', delimiter=' ', header=None, names=['col1', 'col2', 'col3', 'col4', 'col5', 'col6', 'col7', 'col8', 'col9'])
data20 = pd.read_csv('GM_EWPT_Scan20.dat', delimiter=' ', header=None, names=['col1', 'col2', 'col3', 'col4', 'col5', 'col6', 'col7', 'col8', 'col9'])
data21 = pd.read_csv('GM_EWPT_Scan21.dat', delimiter=' ', header=None, names=['col1', 'col2', 'col3', 'col4', 'col5', 'col6', 'col7', 'col8', 'col9'])
data22 = pd.read_csv('GM_EWPT_Scan22.dat', delimiter=' ', header=None, names=['col1', 'col2', 'col3', 'col4', 'col5', 'col6', 'col7', 'col8', 'col9'])
data23 = pd.read_csv('GM_EWPT_Scan23.dat', delimiter=' ', header=None, names=['col1', 'col2', 'col3', 'col4', 'col5', 'col6', 'col7', 'col8', 'col9'])
data24 = pd.read_csv('GM_EWPT_Scan24.dat', delimiter=' ', header=None, names=['col1', 'col2', 'col3', 'col4', 'col5', 'col6', 'col7', 'col8', 'col9'])
data25 = pd.read_csv('GM_EWPT_Scan25.dat', delimiter=' ', header=None, names=['col1', 'col2', 'col3', 'col4', 'col5', 'col6', 'col7', 'col8', 'col9'])
data26 = pd.read_csv('GM_EWPT_Scan26.dat', delimiter=' ', header=None, names=['col1', 'col2', 'col3', 'col4', 'col5', 'col6', 'col7', 'col8', 'col9'])
data27 = pd.read_csv('GM_EWPT_Scan27.dat', delimiter=' ', header=None, names=['col1', 'col2', 'col3', 'col4', 'col5', 'col6', 'col7', 'col8', 'col9'])
data28 = pd.read_csv('GM_EWPT_Scan28.dat', delimiter=' ', header=None, names=['col1', 'col2', 'col3', 'col4', 'col5', 'col6', 'col7', 'col8', 'col9'])
data29 = pd.read_csv('GM_EWPT_Scan29.dat', delimiter=' ', header=None, names=['col1', 'col2', 'col3', 'col4', 'col5', 'col6', 'col7', 'col8', 'col9'])



#合并两个数据
merged_data=pd.concat([data1,data2,data3,data4,data5,data6,data7,data8,data9,data10,data11,data12,data13,data14,data15,data16,data17,data18,data19,data20,data21,data22,data23,data24,data25,data26,data27,data28,data29])

# 筛选出第六列数据不为-1的行
filtered_data = merged_data[merged_data['col6'] != -1]

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
