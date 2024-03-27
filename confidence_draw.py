import numpy as np
import  pandas as pd
from ast import literal_eval
from scipy import stats
import matplotlib.pyplot as plt
from pathlib import Path


import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
bar_colors = ['#845EC2', '#4B4453', '#B0A8B9', '#C34A36', '#FF8066', '#4E8397','#926C00','#009EFA']  # 颜色

class Draw:
    __file_path=None
    __save_name=None
    __fig1_name=None
    __fig2_name=None
    __raw_data=None
    __column=None
    __steps=None
    __conf_data=None
    confidence = 0.95  # 置信区间

    def __init__(self, file_path):
        self.__file_path = Path(file_path)
        print("file path", self.__file_path)
        assert self.__check_path(), f'路径错误，或文件不存在'
        self.__get_names()
        self.__read_data()



    def __check_path(self):
        # 判断文件是否正常
        if not self.__file_path.exists():
            print("文件不存在，路径错误1")
            return False
        else:
            return True

    def __get_names(self):
        self.__save_name = r'./results/'+self.__file_path.stem+"_Confidence_Interval.csv"
        self.__fig1_name= r'./results/'+' '.join(self.__file_path.stem.split("_"))+" Fig1.png"
        self.__fig2_name= r'./results/'+' '.join(self.__file_path.stem.split("_"))+" Fig2.png"


    def __read_data(self):
        self.__raw_data=pd.read_excel(self.__file_path,index_col=0,header=0)

        # 读取excel后格式发生变化
        for column in self.__raw_data.columns.values:
            self.__raw_data[column] = self.__raw_data[column].apply(lambda x: np.array(literal_eval(x)))
        self.__columns = self.__raw_data.columns.values.tolist()
        self.__steps = self.__raw_data.values[0,0].size



    def __get_confidence_interval(self):
        '''
        计算置信区间并保存
        :return:
        '''
        print("计算置信区间")
        index = []
        for step in range(self.__steps):
            index.append("step-" + str(step + 1))
        self.__conf_data = pd.DataFrame({}, columns=self.__columns, index=index)
        for column in self.__columns:
            model=self.__raw_data[column]
            # 将当前列转为二维矩阵形式
            matrix=np.empty([len(model.values), len(model.values[0])])
            for row in range(len(model.values)):
                matrix[row]=model.values[row]

            cells=[]
            for step in range(self.__steps):
                mean=np.mean(matrix[:,step])
                std=np.std(matrix[:,step])

                # 计算置信区间
                z=stats.norm.ppf((1+self.confidence)/2)
                margin_of_error= z*(std/np.sqrt(len(model.values)))
                lower=round(mean-margin_of_error, 6)
                upper=round(mean+margin_of_error, 6)
                cells.append([lower, upper])
            self.__conf_data[column]=cells
        # 保存置信区间结果
        self.__conf_data.to_csv(self.__save_name)

    def __get_model_name(self, column):
        # 从列名中提取模型名
        if (" Train" in column):
            return column.split(" Train")[0]
        elif (" Test" in column):
            return column.split(" Test")[0]
        else:
            print("无法获取模型名，检查列名是否正确")

    def __draw1(self):
        '''
        绘制fig1
        :return:
        '''
        print("绘制Fig1")
        boundary={}
        # 计算左右边界
        for idx, column in enumerate(self.__columns):
            model = self.__conf_data[column]
            matrix = np.empty([len(model.values), len(model.values[0])])
            for row in range(len(model.values)):
                matrix[row] = model.values[row]
            # 计算数据
            boundary[column] = np.mean(matrix, axis=0)

        labels_train = []
        labels_test = []
        means_train = []
        means_test = []
        for key, value in boundary.items():
            if "Train" in key:
                labels_train.append(key)
                means_train.append(round(np.mean(value), 6))
            elif "Test" in key:
                labels_test.append(key)
                means_test.append(round(np.mean(value), 6))
        model_num=len(labels_train)
        labels = labels_train + labels_test
        means = means_train + means_test

        yerr = [round(boundary[label][1] - means[idx], 6) for idx, label in enumerate(labels)]
        x = [i + 0.5 if i < model_num else i + 3.5 for i in range(len(means))]

        groups = 2
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.bar(x, means, width=1, color=bar_colors[0:int(len(labels) / groups)], label=labels)
        ax.errorbar(x, means, yerr, fmt='o', linewidth=1.5, capsize=3, ecolor="black")
        ax.set_xticks([np.mean([x[0 + model_num * i], x[model_num - 1 + model_num * i]]) for i in range(groups)],
                      labels=["Train", "Test"])
        ax.set_ylabel(self.__file_path.stem.split('_')[-1].upper())
        labels = [self.__get_model_name(label) for label in labels]
        ax.legend(labels[0:int(len(labels) / groups)], loc='upper left')
        fig.savefig(self.__fig1_name, dpi=600, bbox_inches='tight')
        print("Fig1 Finished")

    def __draw2(self):
        # 绘制fig2
        print("绘制Fig2")

        # 进获取测试集数据
        columns = [column for column in self.__columns if "Test" in column]
        models_num=len(columns)
        x_ticks = ['step-'+str((step+1)) for step in range(self.__steps)]

        # 用于绘制每个step图形
        # 从筛选数据中获取labels,means,yerr数据
        means = []
        errors = []
        labels = columns * self.__steps

        for idx, label in enumerate(labels):
            step = idx//models_num
            cell=self.__conf_data[label].iloc[step]
            mean=round(np.mean(cell),6)
            error=round((cell[1]-mean),6)
            means.append(mean)
            errors.append(error)

        # 绘图
        x = [i + 0.5 + (i // models_num) * 3 for i in range(len(labels))]

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.bar(x, means, width=1, color=bar_colors[0:models_num], label=labels)
        ax.errorbar(x, means, errors, fmt='o', linewidth=1.5, capsize=3, ecolor="black")
        groups = int(len(x) / models_num)
        x_tixks_loc = [np.mean([x[0 + models_num * i], x[models_num - 1 + models_num * i]]) for i in range(groups)]
        ax.set_xticks(x_tixks_loc, labels=x_ticks)
        ax.set_ylabel(self.__file_path.stem.split('_')[-1].upper())
        labels = [self.__get_model_name(label) for label in labels]
        ax.legend(labels[0:int(len(labels) / 5)], loc='upper left')
        # ax.set_title(name)
        fig.savefig(self.__fig2_name, dpi=600, bbox_inches='tight')
        print("Fig2 finished")

    def start(self):
        self.__get_confidence_interval()
        self.__draw1()
        self.__draw2()


if __name__ == '__main__':
    file_path=r'.\Bitcoin_Multivariate_Mape.xlsx'
    draw=Draw(file_path)
    draw.start()



