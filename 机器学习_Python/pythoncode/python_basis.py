# python ????????????????????????????
# ??????????????????????????????????????????????1??????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
# 2???????????????????????????????????????????????????????????
message="Hello Python!"
print(message)
print(Message)
print(mesage)
_message="Hello Every One!"
print(_message)
1message="Hello Every One!"
print(1message)
message L="Hello Every One"
###########
# python ?????????????????????????????????????????????????1??????????????????/??????????????????????????2????????????????????
a='H'
b="H"
c='Hello!'
c[1:3]
c[0:3] # python ?????0????????????????????????????
# ???????????????????????????????? +????????????????????????????????
d=c+'TEST'
print(d) #?????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????rint
a1=2
a2=3
a3=a1+a2
a4=a1*a2
a5=a1/a2
d=[1,2,3;4,5,6]
# 3???????????????????(liest): ????????????????????????thon?????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
# ???????????????????????????????????????????????????????????????????????????????????????????????????????????????????????[]??????????????????????????py????????????????????????????????????????????????????????????????????

t=['a','b','c','d',a,b,c,d]
t[1:3]
t[4]
t[:4]
t[7]=0
# 4???????????????????(tuple)?????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
t1=('a','b','c','d',a,b,c,d)
print(t1)
t1[7]=0
# 5) ?????????????????????????????????????????dictionary)???????????????ython???????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
## ?????????????????????????????????????????????? ????????????????????????}?????????????????????????????????????????????????????????????????????key)???????????????????????????value?????????????
dict={}
dict['one']='this is one'
dict[2]='this is two'
tinydict={'name': 'runoob','code':6734,'dept':'sales'}
dict[1]
dict['one']
dict[2]
tinydict
tinydict['name']

# ????????????????????????????????????????????????????????????1????????????????????????????????2?????????ype()
## ?????????????
# ?????????????????????????????????tplotlib: cmd, py -m pip install "matplotlib"
import matplotlib.pyplot as plt
squares=[1,4,9,16,25]
fig, ax=plt.subplots()
ax.plot(squares)
plt.show()
# ????????????????????????????????????????
ax.plot(squares,linewidth=3)
ax.set_title("square",fontsize=24)
ax.set_xlabel("x",fontsize=24)
ax.set_ylabel("y",fontsize=14)
ax.tick_params(axis='both', labelsize=14)
plt.show()
# ???????????????????????????????????????????????????????????????????
x=[1,2,3,4,5]
fig, ax=plt.subplots()
ax.plot(x,squares, linewidth=3)
plt.show()

