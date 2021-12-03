from typing import List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

import warnings


class Dataset():
    
    @staticmethod
    def read_train_test(uri: str,
                        test_size: float = 0.10,
                        random_state: int = 42
                       ) -> Tuple[pd.DataFrame, pd.DataFrame]:

        warnings.filterwarnings(action="ignore", message="unclosed", category=ResourceWarning)
        
        # Read training dataset (assumed to fit in memory)
        # as a Tab-Separated-Values (*.tsv) file
        df = pd.read_csv(uri, sep="\t")
            
        # Remove missing values
        df = df.dropna()
        
        # Split dataset into train and test
        train_df, test_df = train_test_split(df,
                                             test_size=test_size,
                                             random_state=random_state)

        return train_df, test_df