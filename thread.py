#-*-encoding=utf-8-*-


'''
使用thread来计算，
如果是计算型的任务GIl会让多线程变慢，效果还不如单线程
'''

import time
import threading


# 装饰器的写法
def text(name):
	def profile(func):
		def wrapper(*args, **kwargs):
			start = time.time()
			res = func(*args, **kwargs)
			end = time.time()
			print('{} cost:{}'.format(name,  end - start))
			return res
		return wrapper
	return profile


def fib(n):
	if n <= 2:
		return 1
	return fib(n-1) + fib(n-2)


@text('nothread')
def nothread():
	fib(35)
	fib(35)
	

@text('hasthread')
def hasthread():
	for i in range(2):
		t = threading.Thread(target=fib, args=(35,))
		t.start()
	main_thread = threading.current_thread()
	for t in threading.enumerate():
		if t is main_thread:
			continue
		t.join()
		

nothread()
hasthread()
'''
nothread cost:6.269474267959595
hasthread cost:5.978232383728027s
'''