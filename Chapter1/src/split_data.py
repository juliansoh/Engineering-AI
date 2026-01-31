def train_test_split(x, y, test_ratio=0.2):
   """
   Split paired data into train/test sets.
   x and y are equal-length 1D lists.
   """
   n = len(x)
   if n != len(y):
      raise ValueError("x and y must be the same length")
   if test_ratio <= 0 or test_ratio >= 1:
      raise ValueError("test_ratio must be between 0 and 1")
   split = int(n * (1 - test_ratio))
   x_train = x[:split]
   x_test = x[split:]
   y_train = y[:split]
   y_test = y[split:]
   return x_train, x_test, y_train, y_test


def test_train_test_split():
   """Test function to verify split lengths and data integrity."""
   # Test data
   x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
   y = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
   test_ratio = 0.2
   
   # Perform split
   x_train, x_test, y_train, y_test = train_test_split(x, y, test_ratio)
   
   # Test 1: Verify lengths
   total_len = len(x)
   expected_train_len = int(total_len * (1 - test_ratio))
   expected_test_len = total_len - expected_train_len
   
   assert len(x_train) == expected_train_len, f"x_train length mismatch: {len(x_train)} != {expected_train_len}"
   assert len(x_test) == expected_test_len, f"x_test length mismatch: {len(x_test)} != {expected_test_len}"
   assert len(y_train) == expected_train_len, f"y_train length mismatch: {len(y_train)} != {expected_train_len}"
   assert len(y_test) == expected_test_len, f"y_test length mismatch: {len(y_test)} != {expected_test_len}"
   
   # Test 2: Verify no data is lost
   assert x_train + x_test == x, "x data lost during split"
   assert y_train + y_test == y, "y data lost during split"
   
   # Test 3: Verify train and test don't overlap
   assert len(set(x_train) & set(x_test)) == 0, "x_train and x_test have overlapping elements"
   assert len(set(y_train) & set(y_test)) == 0, "y_train and y_test have overlapping elements"
   
   # Print lengths
   print(f"Training data length: {len(x_train)}")
   print(f"Test data length: {len(x_test)}")
   print(f"Total data length: {len(x)}")
   print(f"Test ratio: {test_ratio} ({len(x_test)}/{len(x)} = {len(x_test)/len(x):.2f})")
   print("All tests passed!")


if __name__ == "__main__":
   test_train_test_split()