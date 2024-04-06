import pickle
import numpy as np
import gmdh_enum as ge


# region Polynomial

class Polynomial:
    """
    Abstract polynomial
    """

    def __init__(self):
        # Coefficients list
        self.coefficients = None

    @property
    def coefficients_count(self):
        """
        Coefficients count
        """
        return len(self.coefficients) if self.coefficients else 0

    def calculate(self, input_values):
        """
        Calculate polynomial value
        """
        pass


class TwoUnknownPolynomial(Polynomial):
    """
    Abstract polynomial that depends on two unknowns
    """

    def __init__(self):
        super().__init__()

    def calculate_x1x2(self, x1, x2):
        """
        Calculate polynomial value
        """
        return self.calculate([x1, x2])


class LineTwoUnknownPolynomial(TwoUnknownPolynomial):
    """
    Line polynomial that depends on two unknowns

    y = a + b * x1 + c * x2
    """

    def __init__(self, coefficients=None):
        super().__init__()
        self.coefficients = coefficients if coefficients is not None else [0, 0, 0]

    def calculate(self, input_values):
        """
        Calculate polynomial value
        """
        return self.coefficients[0] + self.coefficients[1] * input_values[0] + self.coefficients[2] * input_values[1]


class FirstDegreeTwoUnknownPolynomial(TwoUnknownPolynomial):
    """
    First degree polynomial that depends on two unknowns

    y = a + b * x1 + c * x2 + d * x1 * x2
    """

    def __init__(self, coefficients=None):
        super().__init__()
        self.coefficients = coefficients if coefficients is not None else [0, 0, 0, 0]

    def calculate(self, input_values):
        """
        Calculate polynomial value
        """
        x1, x2 = input_values
        return (self.coefficients[0] + self.coefficients[1] * x1 + self.coefficients[2] * x2 + self.coefficients[3]
                * x1 * x2)


class SecondDegreeTwoUnknownPolynomial(TwoUnknownPolynomial):
    """
    Second degree polynomial that depends on two unknowns

    y = a + b * x1 + c * x2 + d * x1 * x1 + e * x2 * x2 + f * x1 * x2
    """

    def __init__(self, coefficients=None):
        super().__init__()
        self.coefficients = coefficients if coefficients is not None else [0, 0, 0, 0, 0, 0]

    def calculate(self, input_values):
        """
        Calculate polynomial value
        """
        x1, x2 = input_values
        return self.coefficients[0] + self.coefficients[1] * x1 + self.coefficients[2] * x2 + self.coefficients[
            3] * x1 * x1 + self.coefficients[4] * x2 * x2 + self.coefficients[5] * x1 * x2


class ComplexSecondDegreeTwoUnknownPolynomial(TwoUnknownPolynomial):
    """
    Complex second degree polynomial that depends on two unknowns

    y = a + b * x1 + c * x2 + d * x1 * x2 + e * x2 * x1 * x1 + f * x1 * x2 * x2 + g * x1 * x1 * x2 * x2
    """

    def __init__(self, coefficients=None):
        super().__init__()
        self.coefficients = coefficients if coefficients is not None else [0, 0, 0, 0, 0, 0, 0]

    def calculate(self, input_values):
        """
        Calculate polynomial value
        """
        x1, x2 = input_values
        return self.coefficients[0] + self.coefficients[1] * x1 + self.coefficients[2] * x2 + self.coefficients[
            3] * x1 * x2 + self.coefficients[4] * x2 * x1 * x1 + self.coefficients[5] * x1 * x2 * x2 + \
            self.coefficients[6] * x1 * x1 * x2 * x2


# endregion

# region GMDH polynomial


class GmdhPolynomial:
    def __init__(self, polynomial=None):
        """
        Constructor for GMDH Polynomial
        """

        # Index of the first parameter
        self.first_parameter_index = -1

        # Index of the second parameter
        self.second_parameter_index = -1

        # Name of the first parameter
        self.first_parameter_name = ""

        # Name of the second parameter
        self.second_parameter_name = ""

        # Variable level flag
        self.variable_level = False

        # Absolute deviation
        self.abs_deviation = 0.0

        # Relative deviation
        self.rel_deviation = 0.0

        # Mean squared error deviation
        self.mse_deviation = 0.0

        # Value of the polynomial
        self.value = 0.0

        # Polynomial instance
        self.polynomial = polynomial

    @property
    def polynomial_kind(self):
        """
        Determine the type of polynomial
        """
        if isinstance(self.polynomial, LineTwoUnknownPolynomial):
            return ge.GMDHModelKind.LINE_TWO_UNKNOWN_POLYNOMIAL
        elif isinstance(self.polynomial, FirstDegreeTwoUnknownPolynomial):
            return ge.GMDHModelKind.FIRST_DEGREE_TWO_UNKNOWN_POLYNOMIAL
        elif isinstance(self.polynomial, SecondDegreeTwoUnknownPolynomial):
            return ge.GMDHModelKind.SECOND_DEGREE_TWO_UNKNOWN_POLYNOMIAL
        elif isinstance(self.polynomial, ComplexSecondDegreeTwoUnknownPolynomial):
            return ge.GMDHModelKind.COMPLEX_SECOND_DEGREE_TWO_UNKNOWN_POLYNOMIAL
        else:
            raise Exception("Wrong GMDH model kind!!!")

    def calculate_x1x2(self, x1, x2):
        """
        Calculate the value of the polynomial
        """
        return self.polynomial.calculate_x1x2(x1, x2)

    def calculate(self, input_values):
        """
        Calculate the value of the polynomial
        """
        return self.polynomial.calculate(input_values)

    def validate(self):
        """
        Validate the GMDH polynomial
        """
        try:
            # Check for validity
            if self.first_parameter_index < 0 or self.second_parameter_index < 0:
                message = "Wrong polynomial parameter index!"
                return False, message

            if self.variable_level and (not self.first_parameter_name or not self.second_parameter_name):
                message = "Wrong variable name!"
                return False, message

            if not self.polynomial:
                message = "No polynomial instance!"
                return False, message

            return True, ""
        except Exception as e:
            print(f"Model validation error: {e}")
            message = "Not validated!"
            return False, message

# endregion

# region GMDH polynomial matrix


class GmdhPolynomialMatrix:
    def __init__(self):
        """
        Constructor for GMDH Polynomial Matrix
        """
        self._polynomial_matrix = None
        self._default_deviation_kind = ge.DeviationKind.MSE_DEVIATION
        self._parameter_names = None

    @property
    def polynomial_matrix(self):
        """
        Getter for polynomial matrix
        """
        return self._polynomial_matrix

    @polynomial_matrix.setter
    def polynomial_matrix(self, value):
        """
        Setter for polynomial matrix
        """
        self._polynomial_matrix = value
        self._parameter_names = None

    @property
    def default_deviation_kind(self):
        """
        Getter for default deviation kind
        """
        return self._default_deviation_kind

    @default_deviation_kind.setter
    def default_deviation_kind(self, value):
        """
        Setter for default deviation kind
        """
        self._default_deviation_kind = value

    @property
    def parameter_names(self):
        """
        Getter for parameter names
        """
        if self._parameter_names is None:
            self._parameter_names = []

            for i in range(self.polynomial_matrix.shape[1]):
                if self.polynomial_matrix[0, i].variable_level:
                    if (self.polynomial_matrix[0, i].first_parameter_name and
                            (self.polynomial_matrix[0, i].first_parameter_name not in self._parameter_names)):
                        self._parameter_names.append(self.polynomial_matrix[0, i].first_parameter_name)

                    if (self.polynomial_matrix[0, i].second_parameter_name and
                            (self.polynomial_matrix[0, i].second_parameter_name not in self._parameter_names)):
                        self._parameter_names.append(self.polynomial_matrix[0, i].second_parameter_name)

        return self._parameter_names

    @property
    def best_polynomial(self):
        """
        Getter for the best polynomial
        """
        return self.get_best_polynomial(self.default_deviation_kind)

    @property
    def best_polynomial_deviation(self):
        """
        Getter for the deviation of the best polynomial
        """
        return self.get_best_polynomial_deviation(self.default_deviation_kind)

    def validate(self):
        """
        Validate the GMDH Polynomial Matrix
        """
        try:
            if ((self.polynomial_matrix is None) or
                    (self.polynomial_matrix.shape[0] <= 0) or
                    (self.polynomial_matrix.shape[1] <= 0)):
                return False, "Wrong polynomial matrix structure!"

            for i in range(self.polynomial_matrix.shape[0]):
                for j in range(self.polynomial_matrix.shape[1]):
                    polynomial_validate_message = self.polynomial_matrix[i, j].validate()
                    if not polynomial_validate_message[0]:
                        return False, f"Polynomial [{i}, {j}] not valid: {polynomial_validate_message[1]}"

            return True, ""
        except Exception as e:
            print(f"Polynomial model validation error: {e}")
            return False, "Not validated!"

    def calculate_one(self, parameter_values):
        """
        Calculate the polynomial value
        """
        try:
            for i in range(self.polynomial_matrix.shape[0]):
                for j in range(self.polynomial_matrix.shape[1]):
                    if self.polynomial_matrix[i, j].variable_level:
                        self.polynomial_matrix[i, j].value = self.polynomial_matrix[i, j].calculate_x1x2(
                            parameter_values[self.polynomial_matrix[i, j].first_parameter_name],
                            parameter_values[self.polynomial_matrix[i, j].second_parameter_name]
                        )
                    else:
                        self.polynomial_matrix[i, j].value = self.polynomial_matrix[i, j].calculate_x1x2(
                            self.polynomial_matrix[i - 1, self.polynomial_matrix[i, j].first_parameter_index].value,
                            self.polynomial_matrix[i - 1, self.polynomial_matrix[i, j].second_parameter_index].value
                        )
            return self.best_polynomial.value
        except Exception as e:
            print(f"Calculation one error: {e}")
            return None

    def calculate(self, input_values):
        """
        Calculate the polynomial value for input values

        Parameters:
        - input_values (pandas.DataFrame): DataFrame containing input values

        Returns:
        - result (numpy.ndarray): Resulting array of polynomial values
        """
        try:
            result = np.zeros((input_values.shape[0],))
            for row in range(input_values.shape[0]):
                parameter_values = {}

                for col_name in input_values.columns:
                    parameter_values[col_name] = input_values.loc[row, col_name]

                for i in range(self.polynomial_matrix.shape[0]):
                    for j in range(self.polynomial_matrix.shape[1]):
                        if self.polynomial_matrix[i, j].variable_level:
                            self.polynomial_matrix[i, j].value = self.polynomial_matrix[i, j].calculate_x1x2(
                                parameter_values[self.polynomial_matrix[i, j].first_parameter_name],
                                parameter_values[self.polynomial_matrix[i, j].second_parameter_name]
                            )
                        else:
                            self.polynomial_matrix[i, j].value = self.polynomial_matrix[i, j].calculate_x1x2(
                                self.polynomial_matrix[i - 1, self.polynomial_matrix[i, j].first_parameter_index].value,
                                self.polynomial_matrix[i - 1, self.polynomial_matrix[i, j].second_parameter_index].value
                            )

                result[row] = self.best_polynomial.value
            return result
        except Exception as e:
            print(f"Calculation error: {e}")
            return None

    def get_best_polynomial(self, deviation_kind):
        """
        Get the best polynomial based on deviation kind
        """
        try:
            min_deviation = float('inf')
            result = None
            for i in range(self.polynomial_matrix.shape[1]):
                if deviation_kind == ge.DeviationKind.ABS_DEVIATION:
                    deviation = self.polynomial_matrix[-1, i].abs_deviation
                elif deviation_kind == ge.DeviationKind.REL_DEVIATION:
                    deviation = self.polynomial_matrix[-1, i].rel_deviation
                elif deviation_kind == ge.DeviationKind.MSE_DEVIATION:
                    deviation = self.polynomial_matrix[-1, i].mse_deviation
                else:
                    deviation = float('inf')
                if deviation < min_deviation:
                    result = self.polynomial_matrix[-1, i]
                    min_deviation = deviation
            return result
        except Exception as e:
            print(f"Can't get best polynomial: {e}")
            return None

    def get_best_polynomial_deviation(self, deviation_kind):
        """
        Get the deviation of the best polynomial based on deviation kind
        """
        try:
            best_polynomial = self.get_best_polynomial(deviation_kind)
            if deviation_kind == ge.DeviationKind.ABS_DEVIATION:
                return best_polynomial.abs_deviation
            elif deviation_kind == ge.DeviationKind.REL_DEVIATION:
                return best_polynomial.rel_deviation
            elif deviation_kind == ge.DeviationKind.MSE_DEVIATION:
                return best_polynomial.mse_deviation
            else:
                return float('inf')
        except Exception as e:
            print(f"Can't get best polynomial deviation: {e}")
            return float('inf')

    def save_to_file(self, filename):
        """
        Save the polynomial matrix to a file
        """
        try:
            with open(filename, 'wb') as file:
                pickle.dump(self._polynomial_matrix, file)
            return True
        except Exception as e:
            print(f"Error saving polynomial matrix to file: {e}")
            return False

    def load_from_file(self, filename):
        """
        Load the polynomial matrix from a file
        """
        try:
            with open(filename, 'rb') as file:
                self._polynomial_matrix = pickle.load(file)
            return True
        except Exception as e:
            print(f"Error loading polynomial matrix from file: {e}")
            return False

    def __str__(self):
        """
        Returns a string representation of the GMDH model.

        :return: String representation.
        """

        # Prepare description
        description = "Regression model that was built by the GMDH algorithm.\n\n"
        description += "Depends on parameters:\n\n"

        for parameter_name in self.parameter_names:
            description += f" - {parameter_name}\n"

        description += "\n"
        description += f"Selection levels: {self._polynomial_matrix.shape[0]}\n\n"
        description += f"Absolute deviation: {self.get_best_polynomial_deviation(ge.DeviationKind.ABS_DEVIATION)}\n"
        description += f"Relative deviation: {self.get_best_polynomial_deviation(ge.DeviationKind.REL_DEVIATION)}\n"
        description += f"Mean squared deviation: {self.get_best_polynomial_deviation(ge.DeviationKind.MSE_DEVIATION)}\n"

        return description

    def __repr__(self):
        """
        Returns a string representation of the rGMDH model.

        :return: String representation.
        """
        return str(self)

# endregion
