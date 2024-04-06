import numpy as np
import pandas as pd
import gmdh_enum as ge
import gmdh_polynomial as gpol


class GMDH:
    def __init__(self,
                 data_matrix=None,
                 data_vector=None,
                 data_matrix_split_type=ge.DataMatrixSplitType.HALF_AND_HALF_SPLIT,
                 train_percents=50,
                 model_selection_criterion=ge.DeviationKind.MSE_DEVIATION,
                 stop_criterion=ge.DeviationKind.MSE_DEVIATION_QUEUE,
                 model_kind=ge.GMDHModelKind.LINE_TWO_UNKNOWN_POLYNOMIAL,
                 selection_level_count=100,
                 next_level_selection_models_count=3):
        self._data_matrix = data_matrix
        self._data_vector = data_vector
        self._data_matrix_split_type = data_matrix_split_type
        self._train_percents = train_percents
        self._model_selection_criterion = model_selection_criterion
        self._stop_criterion = stop_criterion
        self._model_kind = model_kind
        self._epoch_count = 0
        self._current_epoch_polynomial_vector = None
        self._train_vector = None
        self._test_vector = None
        self._train_matrix = None
        self._test_matrix = None
        self._gmdh_polynomial_matrix = None
        self._selection_level_count = selection_level_count
        self._next_level_selection_models_count = next_level_selection_models_count

    @property
    def gmdh_model(self):
        """
        GMDH model instance
        """
        return self._gmdh_polynomial_matrix

    @gmdh_model.setter
    def gmdh_model(self, value):
        self._gmdh_polynomial_matrix = value

    @property
    def selection_level_count(self):
        """
        Maximum count of selection levels
        """
        return self._selection_level_count

    @selection_level_count.setter
    def selection_level_count(self, value):
        self._selection_level_count = value

    @property
    def next_level_selection_models_count(self):
        """
        Count of model that will be moved to the next selection level
        """
        return self._next_level_selection_models_count

    @next_level_selection_models_count.setter
    def next_level_selection_models_count(self, value):
        self._next_level_selection_models_count = value

    @property
    def data_matrix_split_type(self):
        """
        Matrix split type
        """
        return self._data_matrix_split_type

    @data_matrix_split_type.setter
    def data_matrix_split_type(self, value):
        self._data_matrix_split_type = value

    @property
    def train_percents(self):
        """
        Learning subsequence percents
        """
        return self._train_percents

    @train_percents.setter
    def train_percents(self, value):
        self._train_percents = value

    @property
    def model_selection_criterion(self):
        """
        Model selection criterion
        """
        return self._model_selection_criterion

    @model_selection_criterion.setter
    def model_selection_criterion(self, value):
        self._model_selection_criterion = value

    @property
    def stop_criterion(self):
        """
        Criterion for stopping the learning process
        """
        return self._stop_criterion

    @stop_criterion.setter
    def stop_criterion(self, value):
        self._stop_criterion = value

    @property
    def model_kind(self):
        """
        Model kind
        """
        return self._model_kind

    @model_kind.setter
    def model_kind(self, value):
        self._model_kind = value

    @property
    def goal_name(self):
        """
        Get goal column name
        """
        return self._data_vector.name

    @property
    def data_vector(self):
        """
        Current data vector
        """
        return self._data_vector

    @data_vector.setter
    def data_vector(self, value):
        self._data_vector = value
        self._train_vector = None
        self._test_vector = None

    @property
    def data_matrix(self):
        """
        Current data matrix
        """
        return self._data_matrix

    @data_matrix.setter
    def data_matrix(self, value):
        self._data_matrix = value
        self._train_matrix = None
        self._test_matrix = None

    @property
    def train_vector(self):
        """
        Current learning vector
        """
        if self._train_vector is None:
            self._train_vector = self.get_train_vector()
        return self._train_vector

    @train_vector.setter
    def train_vector(self, value):
        self._train_vector = value

    @property
    def test_vector(self):
        """
        Current checking vector
        """
        if self._test_vector is None:
            self._test_vector = self.get_test_vector()
        return self._test_vector

    @test_vector.setter
    def test_vector(self, value):
        self._test_vector = value

    @property
    def train_matrix(self):
        """
        Current learning vector
        """
        if self._train_matrix is None:
            self._train_matrix = self.get_train_matrix()
        return self._train_matrix

    @train_matrix.setter
    def train_matrix(self, value):
        self._train_matrix = value

    @property
    def test_matrix(self):
        """
        Current checking vector
        """
        if self._test_matrix is None:
            self._test_matrix = self.get_test_matrix()
        return self._test_matrix

    @test_matrix.setter
    def test_matrix(self, value):
        self._test_matrix = value

    def fit(self):
        """
        Run the GMDH algorithm
        """
        self.reset()
        self.gmdh_model = gpol.GmdhPolynomialMatrix()
        self.gmdh_model.polynomial_matrix = np.empty((0, 0))
        try:
            prev_deviation = float('inf')
            current_deviation = float('inf')

            while True:
                if self.build_one_selection_level():
                    # Determine current deviation based on the stop criterion
                    if self.stop_criterion == ge.DeviationKind.ABS_DEVIATION_BEST_MODEL:
                        current_deviation = self.get_current_best_polynomial_deviation(ge.DeviationKind.ABS_DEVIATION)
                    elif self.stop_criterion == ge.DeviationKind.REL_DEVIATION_BEST_MODEL:
                        current_deviation = self.get_current_best_polynomial_deviation(ge.DeviationKind.REL_DEVIATION)
                    elif self.stop_criterion == ge.DeviationKind.MSE_DEVIATION_BEST_MODEL:
                        current_deviation = self.get_current_best_polynomial_deviation(ge.DeviationKind.MSE_DEVIATION)
                    elif self.stop_criterion == ge.DeviationKind.ABS_DEVIATION_QUEUE:
                        current_deviation = self.get_current_epoch_polynomial_vector_deviation(
                            ge.DeviationKind.ABS_DEVIATION)
                    elif self.stop_criterion == ge.DeviationKind.REL_DEVIATION_QUEUE:
                        current_deviation = self.get_current_epoch_polynomial_vector_deviation(
                            ge.DeviationKind.REL_DEVIATION)
                    elif self.stop_criterion == ge.DeviationKind.MSE_DEVIATION_QUEUE:
                        current_deviation = self.get_current_epoch_polynomial_vector_deviation(
                            ge.DeviationKind.MSE_DEVIATION)

                    deviation_valid = current_deviation < prev_deviation
                    prev_deviation = current_deviation

                    if self._epoch_count >= self._selection_level_count or not deviation_valid:
                        break
                else:
                    self.reset()
                    return False

                # Check if the epoch count has reached the selection level count or if the deviation is not valid
                if self._epoch_count >= self._selection_level_count or not deviation_valid:
                    if not deviation_valid:
                        self.gmdh_model.polynomial_matrix = self.gmdh_model.polynomial_matrix[:-1]

            return True
        except Exception as e:
            print(f"Model fit error: {e}")
            self.reset()
            return False

    def predict_one(self, parameter_values):
        """
            Calculate the GMDH model value
        """
        return self.gmdh_model.calculate_one(parameter_values)

    def predict(self, input_values):
        """
            Calculate the GMDH model valus
        """
        return self.gmdh_model.calculate(input_values)

    def reset(self):
        """
        Reset learning process
        """
        self._epoch_count = 0
        self._current_epoch_polynomial_vector = None
        self._train_vector = None
        self._test_vector = None
        self._train_matrix = None
        self._test_matrix = None
        self._gmdh_polynomial_matrix = None

    def build_one_selection_level(self):
        """
            Build one level of GMDH polynomials
        """
        try:
            if (self._epoch_count > 0) and (not self.rebuild_train_and_test_matrices()):
                return False

            self._current_epoch_polynomial_vector = self.build_polynomial_vector(self._epoch_count == 0)
            current_epoch_best_polynomial_vector = self.get_current_epoch_best_polynomial_vector(
                self.model_selection_criterion)

            if self._epoch_count == 0:
                if self.gmdh_model.polynomial_matrix.shape[1] != len(current_epoch_best_polynomial_vector):
                    self.gmdh_model.polynomial_matrix = np.resize(self.gmdh_model.polynomial_matrix,
                                                                  (self.gmdh_model.polynomial_matrix.shape[0],
                                                                   len(current_epoch_best_polynomial_vector)))

                self.gmdh_model.polynomial_matrix = np.vstack((self.gmdh_model.polynomial_matrix,
                                                               current_epoch_best_polynomial_vector))
            else:
                self.gmdh_model.polynomial_matrix = np.vstack((self.gmdh_model.polynomial_matrix,
                                                               current_epoch_best_polynomial_vector))

            self.gmdh_model.default_deviation_kind = self.model_selection_criterion
            self._epoch_count += 1

            return True
        except Exception as e:
            print(f"Build one level of GMDH polynomials error: {e}")
            return False

    def rebuild_train_and_test_matrices(self):
        """
        Rebuild train and test sequences
        """
        try:
            current_epoch_best_polynomial_vector = self.get_current_epoch_best_polynomial_vector(
                self.model_selection_criterion)

            tmp_train_matrix = pd.DataFrame(
                np.zeros((self.train_matrix.shape[0], current_epoch_best_polynomial_vector.shape[0])))

            for j in range(tmp_train_matrix.shape[1]):
                for i in range(tmp_train_matrix.shape[0]):
                    polynomial = current_epoch_best_polynomial_vector[j]

                    tmp_train_matrix.iat[i, j] = polynomial.calculate_x1x2(
                        self.train_matrix.iat[i, polynomial.first_parameter_index],
                        self.train_matrix.iat[i, polynomial.second_parameter_index]
                    )

            self.train_matrix = tmp_train_matrix

            tmp_test_matrix = pd.DataFrame(
                np.zeros((self.test_matrix.shape[0], current_epoch_best_polynomial_vector.shape[0])))

            for j in range(tmp_test_matrix.shape[1]):
                for i in range(tmp_test_matrix.shape[0]):
                    polynomial = current_epoch_best_polynomial_vector[j]

                    tmp_test_matrix.iat[i, j] = polynomial.calculate_x1x2(
                        self.test_matrix.iat[i, polynomial.first_parameter_index],
                        self.test_matrix.iat[i, polynomial.second_parameter_index]
                    )

            self.test_matrix = tmp_test_matrix

            return True
        except Exception as e:
            print(f"Rebuild train and test sequences error: {e}")
            return False

    def get_current_epoch_best_polynomial_vector(self, deviation_kind):
        """
        Get the current epoch's best polynomial vector based on the deviation kind
        """
        try:
            result = np.array([])
            result_indexes = np.array([])

            if self._next_level_selection_models_count <= 0:
                self._next_level_selection_models_count = self.train_matrix.shape[1]

            for n in range(self._next_level_selection_models_count):
                min_deviation = float('inf')
                min_index = -1
                for i in range(len(self._current_epoch_polynomial_vector)):
                    if deviation_kind == ge.DeviationKind.ABS_DEVIATION:
                        if (self._current_epoch_polynomial_vector[i].abs_deviation < min_deviation) and (
                                i not in result_indexes):
                            min_index = i
                            min_deviation = self._current_epoch_polynomial_vector[i].abs_deviation

                    elif deviation_kind == ge.DeviationKind.REL_DEVIATION:
                        if (self._current_epoch_polynomial_vector[i].rel_deviation < min_deviation) and (
                                i not in result_indexes):
                            min_index = i
                            min_deviation = self._current_epoch_polynomial_vector[i].rel_deviation

                    elif deviation_kind == ge.DeviationKind.MSE_DEVIATION:
                        if (self._current_epoch_polynomial_vector[i].mse_deviation < min_deviation) and (
                                i not in result_indexes):
                            min_index = i
                            min_deviation = self._current_epoch_polynomial_vector[i].mse_deviation

                result_indexes = np.append(result_indexes, min_index)
                result = np.append(result, self._current_epoch_polynomial_vector[min_index])

            return result
        except Exception as e:
            print(f"Get the current epoch's best polynomial vector error: {e}")
            return None

    def build_polynomial_vector(self, variable_level):
        """
        Build polynomial vector for the current epoch

        Args:
        - variable_level (bool): variable level for the first epoch

        Returns:
        - polynomial_vector: polynomial vector
        """
        try:
            polynomial_vector = []

            for i in range(self.train_matrix.shape[1]):
                for j in range(i + 1, self.train_matrix.shape[1]):
                    initial_matrix = build_initial_matrix(self.train_matrix, self.model_kind, i, j)
                    initial_vector = np.copy(self.train_vector)

                    n_x, n_y = norm(initial_matrix, initial_vector)
                    root_vector = np.linalg.solve(n_x, n_y)

                    if self.model_kind == ge.GMDHModelKind.LINE_TWO_UNKNOWN_POLYNOMIAL:
                        gmdh_polynomial = gpol.GmdhPolynomial(gpol.LineTwoUnknownPolynomial(root_vector))

                    elif self.model_kind == ge.GMDHModelKind.FIRST_DEGREE_TWO_UNKNOWN_POLYNOMIAL:
                        gmdh_polynomial = gpol.GmdhPolynomial(gpol.FirstDegreeTwoUnknownPolynomial(root_vector))

                    elif self.model_kind == ge.GMDHModelKind.SECOND_DEGREE_TWO_UNKNOWN_POLYNOMIAL:
                        gmdh_polynomial = gpol.GmdhPolynomial(gpol.SecondDegreeTwoUnknownPolynomial(root_vector))

                    elif self.model_kind == ge.GMDHModelKind.COMPLEX_SECOND_DEGREE_TWO_UNKNOWN_POLYNOMIAL:
                        gmdh_polynomial = gpol.GmdhPolynomial(gpol.ComplexSecondDegreeTwoUnknownPolynomial(root_vector))

                    else:
                        return None

                    gmdh_polynomial.first_parameter_index = i
                    gmdh_polynomial.second_parameter_index = j
                    gmdh_polynomial.variable_level = variable_level

                    if variable_level:
                        gmdh_polynomial.first_parameter_name = self.train_matrix.columns[i]
                        gmdh_polynomial.second_parameter_name = self.train_matrix.columns[j]

                    for r in range(self.test_matrix.shape[0]):
                        polynomial_val = gmdh_polynomial.calculate_x1x2(self.test_matrix.iloc[r,
                                                                        gmdh_polynomial.first_parameter_index],
                                                                        self.test_matrix.iloc[r,
                                                                        gmdh_polynomial.second_parameter_index])

                        gmdh_polynomial.abs_deviation += abs(polynomial_val - self.test_vector.iloc[r])
                        if abs(self.test_vector.iloc[r]) != 0:
                            gmdh_polynomial.rel_deviation += (abs(polynomial_val - self.test_vector.iloc[r]) /
                                                              abs(self.test_vector.iloc[r])) * 100.0

                    gmdh_polynomial.abs_deviation /= self.test_matrix.shape[0]
                    gmdh_polynomial.rel_deviation /= self.test_matrix.shape[0]

                    for r in range(self.test_matrix.shape[0]):
                        polynomial_val = gmdh_polynomial.calculate_x1x2(self.test_matrix.iloc[r,
                                                                        gmdh_polynomial.first_parameter_index],
                                                                        self.test_matrix.iloc[r,
                                                                        gmdh_polynomial.second_parameter_index])

                        gmdh_polynomial.mse_deviation += ((polynomial_val - gmdh_polynomial.abs_deviation) *
                                                          (polynomial_val - gmdh_polynomial.abs_deviation))

                    gmdh_polynomial.mse_deviation = np.sqrt(gmdh_polynomial.mse_deviation / self.test_matrix.shape[0])
                    polynomial_vector.append(gmdh_polynomial)

            return polynomial_vector
        except Exception as e:
            print(f"Build polynomial vector error: {e}")
            return None

    def get_current_best_polynomial_deviation(self, deviation_kind):
        """
        Best polynomial deviation
        """
        try:
            current_best_polynomial = self.get_current_best_polynomial(deviation_kind)

            if deviation_kind == ge.DeviationKind.ABS_DEVIATION:
                return current_best_polynomial.abs_deviation
            elif deviation_kind == ge.DeviationKind.REL_DEVIATION:
                return current_best_polynomial.rel_deviation
            elif deviation_kind == ge.DeviationKind.MSE_DEVIATION:
                return current_best_polynomial.mse_deviation
            else:
                return float('inf')  # Return positive infinity for unknown deviation kind
        except Exception as e:
            print(f"Get best polynomial deviation error: {e}")
            return float('inf')  # Return positive infinity in case of any exception

    def get_current_epoch_polynomial_vector_deviation(self, deviation_kind):
        """
        Current epoch polynomial vector deviation
        """
        try:
            deviation = 0.0
            for polynomial in self._current_epoch_polynomial_vector:
                if deviation_kind == ge.DeviationKind.ABS_DEVIATION:
                    deviation += polynomial.abs_deviation
                elif deviation_kind == ge.DeviationKind.REL_DEVIATION:
                    deviation += polynomial.rel_deviation
                elif deviation_kind == ge.DeviationKind.MSE_DEVIATION:
                    deviation += polynomial.mse_deviation

            deviation /= len(self._current_epoch_polynomial_vector)

            return deviation
        except Exception as e:
            print(f"Get current epoch polynomial vector deviation error: {e}")
            return float('inf')

    def get_current_best_polynomial(self, deviation_kind):
        """
        Current best polynomial
        """
        try:
            min_deviation = float('inf')
            result = None
            current_epoch_best_polynomial_vector = self.get_current_epoch_best_polynomial_vector(
                self.model_selection_criterion)

            for polynomial in current_epoch_best_polynomial_vector:
                if deviation_kind == ge.DeviationKind.ABS_DEVIATION:
                    if polynomial.abs_deviation < min_deviation:
                        result = polynomial
                        min_deviation = polynomial.abs_deviation
                elif deviation_kind == ge.DeviationKind.REL_DEVIATION:
                    if polynomial.rel_deviation < min_deviation:
                        result = polynomial
                        min_deviation = polynomial.rel_deviation
                elif deviation_kind == ge.DeviationKind.MSE_DEVIATION:
                    if polynomial.mse_deviation < min_deviation:
                        result = polynomial
                        min_deviation = polynomial.mse_deviation

            return result
        except Exception as e:
            print(f"Get current best polynomial error: {e}")
            return None

    def get_train_vector(self):
        """
        Learning goal vector
        """
        try:
            if self._data_matrix_split_type == ge.DataMatrixSplitType.HALF_AND_HALF_SPLIT:
                half_size = len(self._data_vector) // 2
                res = self._data_vector[:half_size]
                return res

            elif self._data_matrix_split_type == ge.DataMatrixSplitType.PAIRED_UNPAIRED_SPLIT:
                res = self._data_vector[self._data_vector.index % 2 == 0]
                return res

            elif self._data_matrix_split_type == ge.DataMatrixSplitType.PERCENTAGE_SPLIT:
                if (self._train_percents <= 0) or (self._train_percents >= 100):
                    return None

                split_percent = self._train_percents / 100
                split_index = int(len(self._data_vector) * split_percent)
                res = self._data_vector[:split_index]
                return res
            else:
                return None
        except Exception as e:
            print(f"Get train vector error: {e}")
            return None

    def get_test_vector(self):
        """
        Checking goal vector
        """
        try:
            if self._data_matrix_split_type == ge.DataMatrixSplitType.HALF_AND_HALF_SPLIT:
                half_size = len(self._data_vector) // 2
                res = self._data_vector[half_size:]
                return res

            elif self._data_matrix_split_type == ge.DataMatrixSplitType.PAIRED_UNPAIRED_SPLIT:
                res = self._data_vector[self._data_vector.index % 2 == 1]
                return res

            elif self._data_matrix_split_type == ge.DataMatrixSplitType.PERCENTAGE_SPLIT:
                if (self._train_percents <= 0) or (self._train_percents >= 100):
                    return None

                split_percent = self._train_percents / 100
                split_index = int(len(self._data_vector) * split_percent)
                res = self._data_vector[split_index:]
                return res
            else:
                return None
        except Exception as e:
            print(f"Get test vector error: {e}")
            return None

    def get_train_matrix(self):
        """
        Learning dependencies matrix
        """
        try:
            if self._data_matrix_split_type == ge.DataMatrixSplitType.HALF_AND_HALF_SPLIT:
                half_index = len(self._data_matrix) // 2
                res = self._data_matrix.iloc[half_index:]
                return res

            elif self._data_matrix_split_type == ge.DataMatrixSplitType.PAIRED_UNPAIRED_SPLIT:
                res = self._data_matrix[self._data_matrix.index % 2 == 1]
                return res

            elif self._data_matrix_split_type == ge.DataMatrixSplitType.PERCENTAGE_SPLIT:
                if (self._train_percents <= 0) or (self._train_percents >= 100):
                    return None

                split_index = int(len(self._data_matrix) * self._train_percents / 100)
                res = self._data_matrix.iloc[split_index:]
                return res

            else:
                return None
        except Exception as e:
            print(f"Get train matrix error: {e}")
            return None

    def get_test_matrix(self):
        """
        Checking dependencies matrix
        """
        try:
            if self._data_matrix_split_type == ge.DataMatrixSplitType.HALF_AND_HALF_SPLIT:
                half_index = len(self._data_matrix) // 2
                res = self._data_matrix.iloc[:half_index]
                return res

            elif self._data_matrix_split_type == ge.DataMatrixSplitType.PAIRED_UNPAIRED_SPLIT:
                res = self._data_matrix[self._data_matrix.index % 2 == 0]
                return res

            elif self._data_matrix_split_type == ge.DataMatrixSplitType.PERCENTAGE_SPLIT:
                if (self._train_percents <= 0) or (self._train_percents >= 100):
                    return None

                split_index = int(len(self._data_matrix) * self._train_percents / 100)
                res = self._data_matrix.iloc[:split_index]
                return res

            else:
                return None
        except Exception as e:
            print(f"Get test matrix error: {e}")
            return None


def norm(matrix, vector):
    """
    Matrix and vector normalization
    """
    try:
        if vector is not None and vector.size != matrix.shape[0]:
            return None, None

        row_count, col_count = matrix.shape

        result_matrix = np.zeros((col_count, col_count))
        result_vector = np.zeros(col_count)

        for k in range(col_count):
            temp_matrix = matrix.copy()
            temp_vector = vector.copy() if vector is not None else None

            for i in range(row_count):
                multiplier = temp_matrix[i, k]

                for j in range(col_count):
                    temp_matrix[i, j] *= multiplier

                temp_vector[i] *= multiplier

            for j in range(col_count):
                _sum = 0.0
                for i in range(row_count):
                    _sum += temp_matrix[i, j]

                result_matrix[k, j] = _sum / row_count

            _sum = 0.0
            for i in range(row_count):
                _sum += temp_vector[i]

            result_vector[k] = _sum / row_count

        return result_matrix, result_vector
    except Exception as e:
        print(f"Normalization error: {e}")
        return None, None

def build_initial_matrix(source_matrix, model_kind, param_1, param_2):
    """
    Build initial matrix for some type of polynomial model
    """
    try:
        if model_kind == ge.GMDHModelKind.LINE_TWO_UNKNOWN_POLYNOMIAL:
            result = np.zeros((source_matrix.shape[0], 3))

            for i in range(source_matrix.shape[0]):
                result[i, 0] = 1
                result[i, 1] = source_matrix.iloc[i, param_1]
                result[i, 2] = source_matrix.iloc[i, param_2]

            return result

        elif model_kind == ge.GMDHModelKind.FIRST_DEGREE_TWO_UNKNOWN_POLYNOMIAL:
            result = np.zeros((source_matrix.shape[0], 4))

            for i in range(source_matrix.shape[0]):
                result[i, 0] = 1
                result[i, 1] = source_matrix.iloc[i, param_1]
                result[i, 2] = source_matrix.iloc[i, param_2]
                result[i, 3] = source_matrix.iloc[i, param_1] * source_matrix.iloc[i, param_2]

            return result

        elif model_kind == ge.GMDHModelKind.SECOND_DEGREE_TWO_UNKNOWN_POLYNOMIAL:
            result = np.zeros((source_matrix.shape[0], 6))

            for i in range(source_matrix.shape[0]):
                result[i, 0] = 1
                result[i, 1] = source_matrix.iloc[i, param_1]
                result[i, 2] = source_matrix.iloc[i, param_2]
                result[i, 3] = source_matrix.iloc[i, param_1] * source_matrix.iloc[i, param_1]
                result[i, 4] = source_matrix.iloc[i, param_2] * source_matrix.iloc[i, param_2]
                result[i, 5] = source_matrix.iloc[i, param_1] * source_matrix.iloc[i, param_2]

            return result

        elif model_kind == ge.GMDHModelKind.COMPLEX_SECOND_DEGREE_TWO_UNKNOWN_POLYNOMIAL:
            result = np.zeros((source_matrix.shape[0], 7))

            for i in range(source_matrix.shape[0]):
                result[i, 0] = 1
                result[i, 1] = source_matrix.iloc[i, param_1]
                result[i, 2] = source_matrix.iloc[i, param_2]
                result[i, 3] = source_matrix.iloc[i, param_1] * source_matrix.iloc[i, param_2]
                result[i, 4] = source_matrix.iloc[i, param_2] * source_matrix.iloc[i, param_1] * source_matrix.iloc[
                    i, param_1]
                result[i, 5] = source_matrix.iloc[i, param_1] * source_matrix.iloc[i, param_2] * source_matrix.iloc[
                    i, param_2]
                result[i, 6] = source_matrix.iloc[i, param_1] * source_matrix.iloc[i, param_1] * source_matrix.iloc[
                    i, param_2] * source_matrix.iloc[i, param_2]

            return result

        else:
            return None
    except Exception as e:
        print(f"Build initial matrix error: {e}")
        return None
