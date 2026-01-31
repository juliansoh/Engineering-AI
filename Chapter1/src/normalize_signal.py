def _calculate_mean(signal):
    """
    Calculate the mean of non-None values in the signal.
    
    Args:
        signal: List containing numeric values and potentially None values
        
    Returns:
        float: The mean of non-None values, or 0.0 if no valid values exist
    """
    total = 0.0
    count = 0
    for value in signal:
        if value is not None:
            total += value
            count += 1
    return total / count if count > 0 else 0.0


def _calculate_std(signal, mean_value):
    """
    Calculate the standard deviation of non-None values in the signal.
    
    Args:
        signal: List containing numeric values and potentially None values
        mean_value: The pre-calculated mean of the signal
        
    Returns:
        float: The standard deviation of non-None values, or 1.0 if no valid values exist
    """
    sum_squared_diff = 0.0
    count = 0
    for value in signal:
        if value is not None:
            difference = value - mean_value
            sum_squared_diff += difference * difference
            count += 1
    return (sum_squared_diff / count) ** 0.5 if count > 0 else 1.0


def normalize_signal(signal, clip_value=3.0, eps=1e-8):
    """
    Normalize a 1D sensor signal using z-score normalization with clipping.
    
    The normalization process involves:
    1. Calculating mean and standard deviation of non-None values
    2. Scaling: subtract mean and divide by standard deviation (z-score)
    3. Clipping: limit extreme values to the range [-clip_value, +clip_value]
    4. Replacing None values with 0.0 in the output
    
    Args:
        signal (list): Input signal containing numeric values and optionally None values.
                      Must be a list or list-like sequence.
        clip_value (float): Maximum absolute value for clipped normalized values.
                           Must be positive. Default is 3.0.
        eps (float): Small epsilon value added to standard deviation to prevent
                    division by zero. Must be positive. Default is 1e-8.
    
    Returns:
        list: Normalized signal with same length as input. None values are
              replaced with 0.0, and all other values are normalized and clipped.
    
    Raises:
        AssertionError: If input validation fails
        TypeError: If signal contains non-numeric values (except None)
        
    Examples:
        >>> signal = [1.0, 2.0, 3.0, 4.0, 5.0]
        >>> result = normalize_signal(signal, clip_value=2.0)
        >>> # Result will be approximately [-1.41, -0.71, 0.0, 0.71, 1.41]
        
        >>> signal_with_none = [1.0, None, 3.0, 100.0]
        >>> result = normalize_signal(signal_with_none, clip_value=2.0)
        >>> # None is replaced with 0.0, extreme values are clipped
    """
    # Input validation with assertions
    assert signal is not None, "Signal cannot be None"
    assert hasattr(signal, '__len__') and hasattr(signal, '__iter__'), \
        "Signal must be a sequence (list, tuple, etc.)"
    assert clip_value > 0, f"clip_value must be positive, got {clip_value}"
    assert eps > 0, f"eps must be positive, got {eps}"
    
    # Handle empty input
    signal_length = len(signal)
    if signal_length == 0:
        return []
    
    # Validate signal contains only numeric values or None
    for i, value in enumerate(signal):
        if value is not None:
            assert isinstance(value, (int, float)), \
                f"Signal element at index {i} must be numeric or None, got {type(value).__name__}: {value}"
    
    # Calculate statistics from non-None values
    mean_value = _calculate_mean(signal)
    std_value = _calculate_std(signal, mean_value)
    
    # Normalize and clip the signal
    normalized_signal = []
    for value in signal:
        if value is None:
            # Replace None values with 0.0
            normalized_signal.append(0.0)
        else:
            # Apply z-score normalization
            z_score = (value - mean_value) / (std_value + eps)
            
            # Apply clipping
            clipped_value = max(-clip_value, min(clip_value, z_score))
            normalized_signal.append(clipped_value)
    
    return normalized_signal

# Demonstration examples for normalize_signal
if __name__ == "__main__":
    # Example 1: Basic signal normalization
    print("=== Example 1: Basic Signal ===")
    basic_signal = [1.0, 2.0, 3.0, 4.0, 5.0]
    normalized_basic = normalize_signal(basic_signal, clip_value=2.0)
    print(f"Original signal: {basic_signal}")
    print(f"Normalized signal: {[round(x, 3) for x in normalized_basic]}")
    
    # Example 2: Signal with None values and outliers
    print("\n=== Example 2: Signal with None and Outliers ===")
    complex_signal = [1.0, 2.0, 3.0, None, 4.0, 100.0, -50.0]
    normalized_complex = normalize_signal(complex_signal, clip_value=2.0)
    print(f"Original signal: {complex_signal}")
    print(f"Normalized signal: {[round(x, 3) for x in normalized_complex]}")
    
    # Example 3: Empty and edge cases
    print("\n=== Example 3: Edge Cases ===")
    empty_signal = []
    normalized_empty = normalize_signal(empty_signal)
    print(f"Empty signal: {empty_signal} -> {normalized_empty}")
    
    single_value_signal = [42.0]
    normalized_single = normalize_signal(single_value_signal)
    print(f"Single value: {single_value_signal} -> {[round(x, 3) for x in normalized_single]}")
    
    all_none_signal = [None, None, None]
    normalized_all_none = normalize_signal(all_none_signal)
    print(f"All None values: {all_none_signal} -> {normalized_all_none}")