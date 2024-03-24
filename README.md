# Time series model evaluation

## Train and Test

### Get Dataset

we can use the `read_data` function to get the raw data, and then use `data_transform` function to normalize and de normalize the raw data

**Example：**

```python
# read raw data
data, data_len = read_data(file_path, gold_path, dim_type, use_percentage)

# split the dataset into train set and test set
train_set, test_set = data[train_index], data[test_index]

# normalize the dataset
train_set, scalerstrain =data_trasform(train_set)
```





### Create Model

we can use the `create_model` to get the already defined model。

`create_model(model_name, n_features, n_steps_in, n_steps_out)`

* `model_name`：the already defined model name , such as LSTM, BD-LSTM, etc.
* `n_features`：the model will used the number of the dataset features. Features include maximum price, minimum price, opening price, closing price, trading volume, and added gold price.
* `n_steps_in`，`n_steps_out`：the model in and the model out size.

**Example:**

```python
model_name='LSTM'
n_featues='Multi'
n_steps_in = 5 
n_steps_out = 2 

myModel = create_model(model_name, n_features, n_steps_in, n_steps_out)
```



### Train and Test

we use the `train_and_forecast` function to train the model and predict the result.

**Example：**

```python
train_result, test_result = train_and_forecast(myModel, n_features, dim_type, train_set, test_set, n_steps_in,n_steps_out, epochs)
```





### Evaluation

We calculate RMSE and MAPE indicators based on the model prediction results to analyze the performance of the model.



we use the `eval_result(result, n_steps_out, groud_truth, mode)` function to calc the RMSE and MAPE.

* `result`：the model prediction results.
* `n_steps_out`：the model output size
* `ground_truth`：the ground truth
* `mode`：choose the evaluation method, RMSE or MAPE.



**Example:**

* We can choose to calculate evaluation metrics for normalized data

```python
train_minmax_rmse = eval_result(train_result, n_steps_out, train, 0)
test_minmax_rmse = eval_result(test_result, n_steps_out, test, 0)
```



* Similarly, we can also choose to normalize the predicted results before calculating the relevant indicators.

## Results summary

### Results Summary

1. we read the prediction results during the model training and test. Then summary the useful all-steps information from 30 rounds experiments.

```python
# 读取数据
data_eth=pd.read_excel(r"/kaggle/input/results-v2/Bitcoin_Univariate_Rmse.xlsx",index_col=0,header=0)
# type transformer
for column in data_eth.columns.values:
    data_eth[column]=data_eth[column].apply(lambda x:np.array(literal_eval(x)))
    
data_eth.head()


columns = data_eth.columns.values.tolist()
steps=5  # 预测最大步数
index=[]
for step in range(steps):
    index.append("step-"+str(step+1))
    
# 返回表格
result_summary = pd.DataFrame({},columns=columns,index=index)
confidence=0.95  # the confidenc interval
```



2. save the summary result for subsequent drawing.

```python
result_summary=data_summary(data_eth,result_summary,confidence)
result_summary.to_excel("Bitcoin_Univariate_Rmse_Confidence_interval.xlsx")
```



### Draw

1. Firstly, read the above saved sumary result for drawing.

```python
file_path=r"/kaggle/input/confidence-interval/Eth_Univariate_Rmse_Confidence_interval.xlsx"
data = pd.read_excel(file_path,index_col=0)
data=fix_data(data)

# # 查找需要计算的列
# columns = [ele for ele in data.columns.values.tolist() if "MINMAX" in ele]
# data_draw=data[columns]

data_draw = data
boundary=cal_boundary(data_draw)
boundary
```

2. Secondly, We can draw comparison graphs of various models regarding the RMSE index.

```python
# 数据以及参数
labels,means=calc_draw_data(boundary)
yerr=[round(boundary[label][1]-means[idx],6) for idx,label in enumerate(labels)]
x=[i+0.5 if i<6 else i+3.5 for i in range(len(means))]

bar_colors = ['#845EC2', '#4B4453', '#B0A8B9', '#C34A36','#FF8066','#4E8397']  # 颜色


# name="Bitcoin Univariate Rmse_Fig1"
name=get_name_from_path(file_path)
draw(x,means,yerr,labels,name)
```

3. Finally, we can draw comparison graphs of different models under different output step sizes.

```python
# 获取用于绘图数据
steps=5
model_num=len(data_draw.columns.values.tolist())

# bar_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']  # 颜色
bar_colors = ['#845EC2', '#4B4453', '#B0A8B9', '#C34A36','#FF8066','#4E8397']  # 颜色
x_ticks=['step-1','step-2','step-3','step-4','step-5']
labels,means,yerr=get_data(data_draw,steps)
x=[i+0.5+(i//6)*3 for i in range(len(labels))]

name=get_name_from_path(file_path)
draw2(x,means,yerr,x_ticks,bar_colors,labels,name)
```

