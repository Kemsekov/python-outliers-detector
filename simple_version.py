from functions.utils import *
from functions.build_model import build_model
import matplotlib.pyplot as plt

working_dir = "dataset"
dataset = "трещ.csv"

data,mean,scale = load_dataset(working_dir,dataset)
# xi is parameter from 0 to 1 if 0 it means minimal amount of neurons to use
# 1 is max amount of neurons to use. Usually it is set in range [0;0.5]
# bigger xi makes model more sophisticated and more prone to overfit
xi = 0.1
# percent of test split
test_split = 0.1
# input data length
data_input_split = -1
# count of models to build on each iteration
models_count = 5
# max samples to plot
max_elements = 10


shuffled_data = data.copy()
iteration_errors = []
removed_rows = []

while True:
    np.random.shuffle(shuffled_data)
    # test size dependent on test_split
    test_size = int(len(data)*test_split)
    train = shuffled_data[test_size:]
    test = shuffled_data[:test_size]
    
    models = [build_model(train[:,:data_input_split],train[:,data_input_split:],xi) for i in range(0,models_count)]
    
    # error over all samples
    avg_error = average_prediction_error(data,models,data_input_split)

    # average error of all models over test samples
    avg_model_error = np.average(avg_error)
    iteration_errors.append(avg_model_error)
    print(f"Avg model error: {avg_model_error}")
    
    # plot avg_test_error -> it is total model error
    # ask user for outlier id
    # if is outlier is among first N elements
    #   Remove these outliers from `data`` and rebuilt models
    #   
    # if there is no outliers among first N elements.
    #   increase xi and repeat process

    sorted_indices = np.argsort(-avg_error)[:max_elements]


    X = np.arange(0,max_elements)

    plt.clf()
    plt.gcf().set_size_inches(6,7,forward=True)
    plt.subplot(2,1,1)
    plt.plot(X,avg_error[sorted_indices],marker='.')
    plt.xticks(X, labels=sorted_indices)
    plt.xlabel("outlier id")
    plt.ylabel("prediction error")

    plt.subplot(2,1,2)
    plt.plot(np.arange(0,len(iteration_errors)),iteration_errors,marker='.')
    plt.xlabel("iteration N")
    plt.ylabel("model error")

    plt.ion()
    plt.show()
    try:
        input_str = input("Enter rows to remove: ")
        if str.lower(input_str)=="done": break
        if str.lower(input_str)=="": continue
        row_indices = [int(r) for r in input_str.split(" ")]
        for r in data[row_indices]:
            removed_rows.append(r)
        data = np.delete(data,row_indices,axis=0)
        print(f'removed {input_str}')
    except: continue

columns = pd.read_csv(f"{working_dir}/{dataset}").columns

data_without_outliers = restore_row(data,mean,scale)
removed_rows = restore_row(np.array(removed_rows),mean,scale)

df1 = pd.DataFrame(data_without_outliers, columns=columns)
df2 = pd.DataFrame(removed_rows, columns=columns)

df1.to_csv(f"without_outliers_xi={xi}.csv",header=True,index=False)
df2.to_csv(f"outliers_xi={xi}.csv",header=True,index=False)


