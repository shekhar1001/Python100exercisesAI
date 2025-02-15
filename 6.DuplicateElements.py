# Write a function to remove duplicate elements from a list while maintaining the original order of elements.
def duplicate_elements(arr):
    seen=set()
    result=[]
    for item in arr:
        if item not in seen:
            result.append(item)
            seen.add(item)
    return result

print(duplicate_elements([1,2,3,46,6,5,6,5,3,3,5,6,7]))