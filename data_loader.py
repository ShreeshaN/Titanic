import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer

np.set_printoptions(precision=3, threshold=np.inf)
def loadNclean_data(train_filepath):

    # Load train data and clean
    df = pd.read_csv(train_filepath)

    df = df.drop(['Fare','PassengerId','Name','Ticket','Cabin'],axis=1)

    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode())
    df.loc[df['Embarked']=='S','Embarked'] = 0
    df.loc[df['Embarked']=='C','Embarked'] = 1
    df.loc[df['Embarked']=='Q','Embarked'] = 2

    df.loc[df['Sex'] == 'male', 'Sex'] = 0
    df.loc[df['Sex'] == 'female', 'Sex'] = 1

    df['Age'] = df['Age'].fillna(df['Age'].median())

    data = df.values

    # Shuffle the data for randomness
    np.random.shuffle(data)
    train_out = [one_hot_output_vec_for_label(x, (2, 1)) for x in data[:, 0]]
    train_in = [np.reshape(x,(6,1)) for x in data[:,(1,2,3,4,5,6)]]

    data = list(zip(train_in,train_out))

    return (np.asarray(data[:500])),(np.asarray(data[500:len(data)]))


def loadNclean_dataForTf(datapath):
    df = pd.read_csv(datapath)
    df = df.drop(['PassengerId','Name','Ticket','Cabin'],axis=1)
    df['Fare'] = df['Fare'].fillna(df['Fare'].mean())
    df['Embarked'] = df['Embarked'].fillna('S')
    df= pd.get_dummies(df,columns=['Embarked'])
    df.loc[df['Sex'] == 'male', 'Sex'] = 0
    df.loc[df['Sex'] == 'female', 'Sex'] = 1
    df['Age'] = df['Age'].fillna(df['Age'].mean())
    df = df.iloc[np.random.permutation(len(df))]

    input_lables = list(df.columns.values)
    input_lables.remove('Survived')
    input = df.as_matrix(input_lables)

    out = df.as_matrix(['Survived'])
    output = []
    for x in out:
        output.append(one_hot_output_vec_for_label(x, 2, True))

    share=700
    train_in = input[:share]
    train_out = output[:share]
    test_in = input[share:len(input)]
    test_out = output[share:len(output)]

    return  train_in,train_out,test_in,test_out


def loadNclean_dataForTf_Kaggle(testpath):

    df = pd.read_csv(testpath)
    df = df.drop(['PassengerId','Name','Ticket','Cabin'],axis=1)
    df['Fare'] = df['Fare'].fillna(df['Fare'].mean())
    df['Embarked'] = df['Embarked'].fillna('S')

    cols=['Embarked']
    df = one_hot_for_input(df,cols)

    df.loc[df['Sex'] == 'male', 'Sex'] = 0
    df.loc[df['Sex'] == 'female', 'Sex'] = 1
    df['Age'] = df['Age'].fillna(df['Age'].mean())

    data = df.values
    input = data[:]
    return input


def one_hot_output_vec_for_label(output, shape, forTf=False):
    # Prepare a one hot vector for output values Ex: 2 ->[0,0,1] | 1 ->[0,1,0] | 0 ->[1,0,0]
    if forTf:
        vec = shape * [0]
        vec[output]=1.0
    else:
        vec = np.zeros(shape)
        vec[output] = 1.0
    return vec


def one_hot_for_input(df,cols):
    vec = DictVectorizer()
    vec_data = pd.DataFrame(vec.fit_transform(df[cols].to_dict(orient='records')).toarray())
    vec_data.columns = vec.get_feature_names()
    vec_data.index = df.index
    df = df.drop(cols, axis=1)
    df = df.join(vec_data)
    return df

def prep_submission_file(test_path,predictions):
    df = pd.read_csv(test_path)

    df = df['PassengerId']
    ids = df.values
    data = list(zip(ids,predictions))
    with open(r'sub.csv', 'a') as file:
        file.write("PassengerId" + "," + "Survived" + '\n')
        for x in data:
            file.write(str(x[0]) + "," + str(x[1]) + '\n')

