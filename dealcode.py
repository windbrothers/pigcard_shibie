from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import math
import random
# 参数times用来模拟网络请求的时间
def Sum(num):
    Sum1=0
    for i in range(num):
#        print(num,i)
        Sum1=Sum1+i
    return Sum1
    
def get_html(times):
    time.sleep(times)
    print("第 {}次执行".format(times))
    a=Sum(10)
    t=random.randint(1,10)
    time.sleep(t)
    print(a)
    return a

#executor = ThreadPoolExecutor(max_workers=2000)
#urls = [1, 2, 3, 4, 5, 6,7, 8, 9,10,11,12,13,14,15,16,17,18,19,20] # 并不是真的url
#all_task = [executor.submit(get_html,(url)) for url in urls]
#
#for future in as_completed(all_task):
#    data = future.result()
#    print("Sum{}".format(data))
    
if __name__ == '__main__':
    
    executor = ThreadPoolExecutor(max_workers=2000)
    num=random.randint(1,10)
    all_task = [executor.submit(get_html,num)]
    for future in as_completed(all_task):
        data = future.result()
        print("in main: get page {}s success".format(data))    
