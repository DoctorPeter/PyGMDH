import numpy as np
import pandas as pd
import gmdh_model as g
import gmdh_enum as ge
import gmdh_polynomial as gpol
from sklearn.datasets import fetch_california_housing

california_housing_data = fetch_california_housing()
features = pd.DataFrame(data=california_housing_data.data, columns=california_housing_data.feature_names)
target = pd.Series(california_housing_data.target, name='MEDV')

features = features.iloc[:100]
target = target.iloc[:100]


gmdh = g.GMDH(data_matrix=features,
              data_vector=target,
              data_matrix_split_type=ge.DataMatrixSplitType.PAIRED_UNPAIRED_SPLIT,
              train_percents=50,
              model_selection_criterion=ge.DeviationKind.REL_DEVIATION,
              stop_criterion=ge.DeviationKind.REL_DEVIATION_QUEUE,
              model_kind=ge.GMDHModelKind.COMPLEX_SECOND_DEGREE_TWO_UNKNOWN_POLYNOMIAL,
              selection_level_count=100,
              next_level_selection_models_count=0)

if gmdh.fit():
    print(gmdh.gmdh_model)
    predict_result = gmdh.predict(features)
    demo_matrix = pd.DataFrame(features)
    demo_matrix['MEDV'] = target
    demo_matrix['PREDICT'] = predict_result
    print(demo_matrix.shape)
    print(demo_matrix)
else:
    if gmdh.gmdh_model is not None and gmdh.gmdh_model.polynomial_matrix is not None:
        res, message = gmdh.gmdh_model.polynomial_matrix.validate()
        if res:
            print(gmdh.gmdh_model)
        else:
            print(message)
    else:
        print("No GMDH model")

