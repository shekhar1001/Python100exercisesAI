# def find_second_largest(nums):
#     if len(nums) < 2:
#         return None
    
#     first = second = float('-inf')
#     for num in nums:
#         if num > first:
#             second = first
#             first = num
#         elif first > num > second:
#             second = num
    
#     return second if second != float('-inf') else None

# # Example usage:
# nums = [10, 20, 4, 45, 99]
# print(find_second_largest(nums))  # Output: 45

def finding_second_largest(arr):
    if len(arr)<2:
        return None
    arr=list(set(arr))
    arr.sort(reverse=True)
    return arr[1]

def finding_third_largest(arr):
    if len(arr)<3:
        return None
    arr=list(set(arr))
    arr.sort(reverse=True)
    return arr[2]

print(finding_second_largest([20,100,300,24,60,55]))
print(finding_second_largest([20,100,50,25,1,21,24,56,33]))
print(finding_third_largest([20,100,300,24,60,55]))
print(finding_third_largest([20,100,50,25,1,21,24,56,33]))

