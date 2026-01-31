#Write a function moving_average(x,window) that returns a NumPy array of moving average
import numpy as np
def moving_average(x, window):
    """
    Calculate the moving average of a 1D NumPy array.

    Parameters:
    x (np.ndarray): Input 1D array.
    window (int): The size of the moving window.

    Returns:
    np.ndarray: Array of moving averages.
    """
    if window < 1:
        raise ValueError("Window size must be at least 1.")
    if window > len(x):
        raise ValueError("Window size must not be larger than the input array length.")

    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    moving_avg = (cumsum[window:] - cumsum[:-window]) / window
    return moving_avg

#Simple test that tries a simplay array like np.array([1,2,3,4,5]) with window 3
if __name__ == "__main__":
    test_array = np.array([1, 2, 3, 4, 5])
    window_size = 3
    result = moving_average(test_array, window_size)
    print("Input array:", test_array)
    print("Window size:", window_size)
    print("Moving average:", result)