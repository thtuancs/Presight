def sumOfOddNumbers(a,b):
    start = a + 1 if a % 2 == 0 else a + 2
    end = b - 1 if b % 2 == 0 else b - 2
    if start > end: return 0
    odd_sum = (end+start)*((end-start) // 2 + 1) // 2
    return odd_sum % 10000007

if __name__ == "__main__":
    a = 1
    b = 1238127813
    print(sumOfOddNumbers(a,b))