import numpy as np
import  pandas as pd
from ast import literal_eval
from scipy import stats
import matplotlib.pyplot as plt
from pathlib import Path

# 修改数据
def fix_data(data):
    # 讲读取后的数据恢复成list
    for column in data.columns.values:
        data[column]=data[column].apply(lambda x:np.array(literal_eval(x)))
    return data
def data_summary(data_org,result_summary,confidence):
    # 计算95置信区间以及绘制直方图
    columns=data_org.columns
    for column in columns:
        # 获取模型数据
        model = data_org[column]
        # 将数据转换成二维数组求解
        martix = np.empty((len(model.values), len(model.values[0]))) 
        for row in range(len(model.values)):
            martix[row]=model.values[row]

        cells = []
        for i in range(len(model.values[0])):
            # i列代表step-i
            mean=np.mean(martix[:,i])
            std_dev=np.std(martix[:,i])

            # 计算置信区间  
            z = stats.norm.ppf((1 + confidence) / 2)  
            margin_of_error = z * (std_dev / np.sqrt(len(model.values))) 

            # 创建置信区间  
            lower_bound = round(mean - margin_of_error,4)  
            upper_bound = round(mean + margin_of_error,4)
            cells.append([lower_bound,upper_bound])
        result_summary[column]=cells
    return result_summary

# 修改数据
def fix_data(data):
    # 讲读取后的数据恢复成list
    for column in data.columns.values:
        data[column]=data[column].apply(lambda x:np.array(literal_eval(x)))
    return data
def data_summary(data_org,result_summary,confidence):
    # 计算95置信区间以及绘制直方图
    columns=data_org.columns
    for column in columns:
        # 获取模型数据
        model = data_org[column]
        # 将数据转换成二维数组求解
        martix = np.empty((len(model.values), len(model.values[0]))) 
        for row in range(len(model.values)):
            martix[row]=model.values[row]

        cells = []
        for i in range(len(model.values[0])):
            # i列代表step-i
            mean=np.mean(martix[:,i])
            std_dev=np.std(martix[:,i])

            # 计算置信区间  
            z = stats.norm.ppf((1 + confidence) / 2)  
            margin_of_error = z * (std_dev / np.sqrt(len(model.values))) 

            # 创建置信区间  
            lower_bound = round(mean - margin_of_error,4)  
            upper_bound = round(mean + margin_of_error,4)
            cells.append([lower_bound,upper_bound])
        result_summary[column]=cells
    return result_summary
    
def fix_data(data):
    # 讲读取后的数据恢复成list
    for column in data.columns.values:
        data[column]=data[column].apply(lambda x:np.array(literal_eval(x)))
    return data
def cal_boundary(data):
    # 计算每个模型的左右边界
    boundary={}
    columns=data.columns.values
    for idx,column in enumerate(columns):
        model=data[column]
        matrix=np.empty((len(model.values),len(model.values[0])))
        for row in range(len(model.values)):
            matrix[row]=model.values[row]
        # 计算数据
        boundary[column]=np.mean(matrix,axis=0)
    return boundary

def calc_draw_data(boundary):
    labels_train=[]
    labels_test=[]
    means_train=[]
    means_test=[]
    for key,value in boundary.items():
        if "Train" in key:
            labels_train.append(key)
            means_train.append(round(np.mean(value),4))
        elif "Test" in key:
            labels_test.append(key)
            means_test.append(round(np.mean(value),4))

    labels=labels_train+labels_test
    means=means_train+means_test
    return labels,means

def draw(x,means,yerr,labels, name):
    # 绘图
    model_num=6
    groups=2
    fig,ax = plt.subplots(figsize=(8,6))
    ax.bar(x,means,width=1,color=bar_colors[0:int(len(labels)/2)],label=labels)
    ax.errorbar(x,means,yerr, fmt='o',linewidth=2, capsize=6,ecolor="black")  
    # ax.set_xticks([(ax.get_xticks()[-1])*0.25,(ax.get_xticks()[-1])*0.75],labels=["Train","Test"],fontsize=20)
    ax.set_xticks([np.mean([x[0+model_num*i], x[model_num-1+ model_num*i]]) for i in range(groups)],labels=["Train","Test"],fontsize=20)
    ax.set_ylabel(name.split(' ')[-1],fontsize=15)
    
    labels=[get_model_name(label) for label in labels]
    ax.legend(labels[0:int(len(labels)/2)])
    ax.set_title(name)
    fig.savefig(name+" Fig1.png",dpi=600,bbox_inches='tight')

def get_data(data_draw,steps):
    # 用于绘制每个step图形
    # 从筛选数据中获取labels,means,yerr数据
    means=[]
    yerr=[]
    model_num=len(data_draw.columns.values.tolist())
    labels = data_draw.columns.values.tolist()*steps

    for i in range(len(labels)):
        row=i//model_num
        col=i%model_num
        ele=data_draw.iloc[row,col]
        mean=round(np.mean(ele),6)
        error=round((ele[1]-mean),6)
        means.append(mean)
        yerr.append(error)
    return labels,means,yerr


def draw2(x,means,yerr,x_ticks,bar_colors,labels,name):
    model_num=6
    # 根据step绘图
    fig,ax=plt.subplots(figsize=(10,8))

    ax.bar(x,means,width=1,color=bar_colors,label=labels)
    ax.errorbar(x,means,yerr, fmt='o',linewidth=2, capsize=6,ecolor="black")

    x_tick_len = ax.get_xticks()[-1]
    # x_tixks_loc=[x_tick_len*0.05,x_tick_len*0.25,x_tick_len*0.45,x_tick_len*0.65,x_tick_len*0.85]
    groups=int(len(x)/model_num)
    x_tixks_loc=[np.mean([x[0+model_num*i],x[model_num-1+model_num*i]]) for i in range(groups)]
    ax.set_xticks(x_tixks_loc,labels=x_ticks,fontsize=20)
    ax.set_ylabel(name.split(' ')[-1])
    labels=[get_model_name(label) for label in labels]
    ax.legend(labels[0:int(len(labels)/5)])
    ax.set_title(name)
    fig.savefig(name+" Fig2.png",dpi=600,bbox_inches='tight')
    
def get_model_name(column):
    # 从列名中提取模型名
    
    if(" Train" in column):
        return column.split(" Train")[0]
    elif(" Test" in column):
        return column.split(" Test")[0]
    else:
        print("无法获取模型名，检查列名是否正确")
        
        

def get_name_from_path(file_path): 
    return ' '.join(Path(file_path).stem.split("_")[0:-2])

    

def f1():
    '''
    读取原始数据，求置信区间并保存
    '''
    # 读取数据
    data_eth=pd.read_excel(r"D:\pythonProject\doge_multivariate_mape.xlsx",index_col=0,header=0)
    # 由于读取excel之后数据发生了变化（list变成了str类型），需要转换成list类型
    for column in data_eth.columns.values:
        data_eth[column]=data_eth[column].apply(lambda x:np.array(literal_eval(x)))
        
    data_eth.head()
    print("原始数据：",data_eth.head())
    
    columns = data_eth.columns.values.tolist()
    steps=5  # 预测最大步数
    index=[]
    for step in range(steps):
        index.append("step-"+str(step+1))
    
    # 返回表格
    result_summary = pd.DataFrame({},columns=columns,index=index)
    confidence=0.95  # 置信区间

    result_summary=data_summary(data_eth,result_summary,confidence)
    result_summary.to_excel("doge_multivariate_mape_Confidence_interval.xlsx")


    
def f2():
    '''
    读取置信区间数据，并绘制Fig-1、Fig-2
    '''

    file_path=r"D:\pythonProject\doge_multivariate_mape_Confidence_interval.xlsx"
    
    # Fig-1
    data = pd.read_excel(file_path,index_col=0)
    data=fix_data(data)

    # # 查找需要计算的列
    # columns = [ele for ele in data.columns.values.tolist() if "MINMAX" in ele]
    # data_draw=data[columns]

    data_draw = data
    boundary=cal_boundary(data_draw)
    boundary


    # 数据以及参数
    labels,means=calc_draw_data(boundary)
    yerr=[round(boundary[label][1]-means[idx],6) for idx,label in enumerate(labels)]
    x=[i+0.5 if i<6 else i+3.5 for i in range(len(means))]
    # bar_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']  # 颜色
    bar_colors = ['#845EC2', '#4B4453', '#B0A8B9', '#C34A36','#FF8066','#4E8397']  # 颜色
    # bar_colors = ['#E76254', '#EB8C49', '#F4AB5C', '#FBD071','#FCE7B6','#AEDCE0','#79BBD5','#588EAC','#3F6694','#26456F']  # 颜色

    # name="Bitcoin Univariate Rmse_Fig1"
    name=get_name_from_path(file_path)
    draw(x,means,yerr,labels,name)


    # Fig-2
    # 筛选数据 选取数据MINMAX TEST
    data = pd.read_excel(file_path,index_col=0)
    data=fix_data(data)

    # ------------------
    # 获测试集数据
    columns = [ele for ele in data.columns.values.tolist() if "Test" in ele]
    data_draw=data[columns]
    # -------------------

    # data_draw=data
    # 获取用于绘图数据
    steps=5
    model_num=len(data_draw.columns.values.tolist())

    # bar_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']  # 颜色
    bar_colors = ['#845EC2', '#4B4453', '#B0A8B9', '#C34A36','#FF8066','#4E8397']  # 颜色
    x_ticks=['step-1','step-2','step-3','step-4','step-5']
    labels,means,yerr=get_data(data_draw,steps)
    x=[i+0.5+(i//6)*3 for i in range(len(labels))]


    # name="Bitcoin Univariate Rmse Fig2"
    name=get_name_from_path(file_path)
    draw2(x,means,yerr,x_ticks,bar_colors,labels,name)


def main():
    # f1主要负责读取原始数据，并求置信区间
    f1()
    
    # f2()主要负责读取置信区间数据，并绘制图形
    f2()

if __name__ == '__main__':
    main()


