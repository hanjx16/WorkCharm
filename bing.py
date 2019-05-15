#-*-encoding=utf-8-*-

from functools import partial
from operator import mul

# 数据立即绑定
def generate():
	temp = [lambda x,i=i: i * x for i in range(4)]
	return temp


# 立即绑定的其他写法
def generate1():
	return [partial(mul, i) for i in range(4)]


# 立即绑定写法
def generate2():
	for i in range(4):
		yield lambda x :i * x


#  数据惰性绑定
def generate3():
	temp = [lambda x:i*x for i in range(4)]
	return temp


for a in generate2():
	print(a(2))
