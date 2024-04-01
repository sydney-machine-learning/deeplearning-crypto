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
In this part, we will integrate the result data including the training and testing results during the 30 rounds experiments.
We will calculate the RMSE and MAPE index from the above results data, and then plots Corresponding figures.


The results data formate for input is excel.File Content main include RMSE or MAPE index about all models during training and testing.
The summary and draw corresponding figures functioins are integrated into a `draw.py` python module,so we can import this module to gets the results quickly.


Example:
```python
from draw import Draw
file_path=r'Bitcoin_Multivariate_Mape.xlsx'
draw=Draw(file_path)
draw.start()
```

### Results Summary
We provide the `Draw.start()` api to finish all the operations and get all results.Meanwhile, we also provide some 
extra apis for you to calculate intermediate results for analyse step by step.


We provide `Draw.get_confidence_interval()` function for you to get the confidence interval from the origin input.


Example:
```python
from draw import Draw
file_path=r'Bitcoin_Multivariate_Mape.xlsx'
draw=Draw(file_path)
draw.get_confidence_interval()
```

the calculated results will be saved in `csv` format by default.



### Draw
In this part, we will visualize the confidence interval results by plotting corresponding figures.


We provide an independent API(`draw1()` and `draw2()`)for you to draw corresponding images.

```python
from draw import Draw
file_path=r'Bitcoin_Multivariate_Mape.xlsx'
draw=Draw(file_path)
draw.draw1()

draw.draw2()

```
In the `draw.py` module, we can set the bar_colors parameter to custom display color range.





