value = -102047.3  # The value to normalize
min_value = 0  # The minimum value in the range
max_value = 100  # The maximum value in the range

# Normalize the value
normalized_value = (value - min_value) / (max_value - min_value)

# Print the normalized value
print(normalized_value)


def __normalizeValue(self, minValue, maxValue, value):
    normalized_value = (value - minValue) / (maxValue - minValue)

    return normalized_value