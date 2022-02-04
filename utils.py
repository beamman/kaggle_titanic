from matplotlib.pyplot import axis
import torch
import pandas as pd
import numpy as np


def load_train_dataset():

    
    # Load Data
    train_dataset = pd.read_csv('data/train.csv')

    # Define x (data)
    x = train_dataset.drop(['PassengerId', 'Name', 'Ticket', 'Fare', 'Cabin'], axis=1)
    x['Age'] = x['Age'].fillna(0)
    x['Embarked'] = x['Embarked'].fillna('C')

    # ['Embarked'] S, C, Q -> 1, 2, 3 to train
    embarked = x['Embarked'].values
    for i in range(len(embarked)):
        s, c, q = "S", "C", "Q"

        if s in embarked[i]:
            embarked[i] = 1
        
        elif c in embarked[i]:
            embarked[i] = 2
        
        else:
            embarked[i] = 3
    
    x = x.drop(['Embarked'], axis=1)
    Embarked = np.array(embarked)
    x['Embarked'] = Embarked
    x['Embarked'] = x.Embarked.astype(int)

    # ['Sex'] male, female -> 1, 2
    sex = x['Sex'].values
    for i in range(len(sex)):
        male, female = "male", "female"

        if male in sex[i]:
            sex[i] = 1
        
        else:
            sex[i] = 2
    
    x = x.drop(['Sex'], axis=1)
    Sex = np.array(sex)
    x['Sex'] = sex
    x['Sex'] = x.Sex.astype(int)

    # Dataframe -> tensor
    x = x.values
    x = torch.from_numpy(x)


    # Define y (target)
    y = train_dataset['Survived']
    y = pd.DataFrame(y)

    # Dataframe -> tensor
    y = y.values
    y = torch.from_numpy(y)

    return x, y