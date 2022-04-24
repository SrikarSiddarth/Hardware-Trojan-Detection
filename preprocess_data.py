import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, scale
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import seaborn as sns


def create_numerics(data):
    # Get nominal columns
    nominal_cols = data.select_dtypes(include='object').columns.tolist()
    print('nominal_cols: ', nominal_cols)
    # Turn nominal to numeric
    for nom in nominal_cols:
        enc = LabelEncoder()
        enc.fit(data[nom])
        data[nom] = enc.transform(data[nom])

    return data

    
def prepare_data():
    data = pd.read_excel("aes.xlsx")
    data = data.dropna()

    
    
    # trojan_free = data.loc[data['Label']=="'Trojan Free'"].reset_index()    
    # print('number of trojan free circuits: ',len(trojan_free))
    # # balance the ratio between trojan free and infected of the same circuit category
    # for i in range(len(trojan_free)):
    #     category_substring = trojan_free['Circuit'][i].replace("'",'')
    #     circuit_group = data[data['Circuit'].str.contains(category_substring)]
        
        
    #     df1 = circuit_group.iloc[0:1]
        
    #     if len(circuit_group) > 1:
    #         data = data.append([df1]*(len(circuit_group)-1), ignore_index=True)
    # print('pd.DataFrame(data["Label"]: ',pd.DataFrame(data["Label"]))
    # # print(category_substring)
    # print(data['Circuit'].str.contains(category_substring))
    
    data.drop(columns=['Circuit'], inplace=True)

    # # normalizing data
    # data = scale(data)


    # data = create_numerics(data)

    # print('pd.DataFrame(data["Label"]: ',pd.DataFrame(data["Label"]))
    # data.info()
    # data = shuffle(data, random_state=42)
    # data.info()

    # Create correlation matrix
    # corr_matrix = data.corr().abs()

    # # print('corr_matrix: ',corr_matrix)

    # # Select upper triangle of correlation matrix
    # upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape),
    #                                   k=1).astype(np.bool))

    # # print('upper: ',upper)

    # # Find index of feature columns with correlation greater than 0.95
    # to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]

    # print('to_drop: ',to_drop)

    # # Drop features
    # data = data.drop(data[to_drop], axis=1)

    # drop columns that contains only zeros
    # data = data.loc[:, (data != 0).any(axis=0)]

    # print('data after removing zero columns: ')
    # print(data.info())
    # data.to_excel('modifiedHTData.xlsx')
    data = shuffle(data, random_state=42)

    y = pd.DataFrame(data["Label"]).values
    # print('y: ',np.mean(y))

    x = data.drop(["Label"], axis=1)
    # print('x: ',type(x))

    scaler = MinMaxScaler(feature_range=(0, 1))
    x = scaler.fit_transform(x)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.6, random_state=1)
    
    """
    # plot the correlated features
    sns.heatmap(
        corr_matrix,
        vmin=-1, vmax=1, center=0,
        cmap=sns.diverging_palette(20, 220, n=200),
        square=True
    )
    plt.title("Features correlation")
    plt.show()
    """
    # print(x_train.shape, x_test.shape)
    return(x_train, x_test, y_train, y_test)