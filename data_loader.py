from random import shuffle
import torch
import pandas as pd
import numpy as np

from torch.utils.data import DataLoader, Dataset


class titanicDataset(Dataset):

    def __init__(self, data, labels, flatten=True):
        self.data = data
        self.labels = labels
        self.flatten = flatten

        super().__init__()

    
    def __len__(self):
        return self.data.size(0)

    
    def __get_item__(self, index):
        x = self.data[index]
        y = self.labels[index]

        if self.flatten:
            x = x.view(-1)
        
        return x, y    


def load_titanic(is_train=True):
    if is_train:
        train_dataset = pd.read_csv('data/train.csv')

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

    else:
        test_dataset = pd.read_csv('data/test.csv')

        x = test_dataset.drop(['PassengerId', 'Name', 'Ticket', 'Fare', 'Cabin'], axis=1)
        x['Age'] = x['Age'].fillna(0)

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

        return x
    

def get_loaders():
    x, y = load_titanic(is_train=True)

    train_cnt = int(x.size(0) * .8)
    valid_cnt = x.size(0) - train_cnt

    # shuffle dataset to split train/valid
    indices = torch.randperm(x.size(0))
    train_x, valid_x = torch.index_select(x, dim=0, index=indices).split([train_cnt, valid_cnt], dim=0)
    train_y, valid_y = torch.index_select(y, dim=0, index=indices).split([train_cnt, valid_cnt], dim=0)

    train_loader = DataLoader(dataset=titanicDataset(train_x, train_y, flatten=True), batch_size=32, shuffle=True)
    valid_loader = DataLoader(dataset=titanicDataset(valid_x, valid_y, flatten=True), batch_size=32, shuffle=True)

    return train_loader, valid_loader
