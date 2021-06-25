from functools import wraps
import time

# class logit(object):
#     def __init__(self, logfile='out.log'):
#         self.logfile = logfile
#
#     def __call__(self, func):
#         @wraps(func)
#         def wrapped_function(*args, **kwargs):
#             log_string = func.__name__ + " was called"
#             print(log_string)
#             # 打开logfile并写入
#             with open(self.logfile, 'a') as opened_file:
#                 # 现在将日志打到指定的文件
#                 opened_file.write(log_string + '\n')
#             # 现在，发送一个通知
#             self.notify()
#             return func(*args, **kwargs)
#
#         return wrapped_function
#
#     def notify(self):
#         # logit只打日志，不做别的
#         pass
#
#
# class email_logit(logit):
#     '''
#     一个logit的实现版本，可以在函数调用时发送email给管理员
#     '''
#
#     def __init__(self, email='admin@myproject.com', *args, **kwargs):
#         self.email = email
#         super(email_logit, self).__init__(*args, **kwargs)
#
#     def notify(self):
#         # 发送一封email到self.email
#         # 这里就不做实现了
#         pass

class limit_time(object):
    def __init__(self, max_secend=60, max_n=500):
        self.max_secend = max_secend  #
        self.max_n = max_n
        self.sumtime = 0
        self.n = 0

    def __call__(self, func):
        @wraps(func)
        def wrapped_function(*args, **kwargs):
            log_string = func.__name__ + " was called"
            print('start to timing: ')
            ss = time.time()
            res = func(*args, **kwargs)
            range_time = time.time() - ss
            self.sumtime += range_time
            self.n += 1

            if self.sumtime < self.max_secend and self.n > self.max_n:
                time.sleep(self.max_secend - self.sumtime + 2)
                self.sumtime = 0
                self.n = 0
            if self.sumtime > self.max_secend:
                self.sumtime = 0
                self.n = 0
            self.n += 1
            print(self.n)
            print('waste time: '+str(range_time))
            print('sum time: ' + str(self.sumtime))
            return res

        return wrapped_function

    def notify(self):
        # logit只打日志，不做别的
        pass

class time_count(object):
    def __init__(self):
        self.sumtime = 0
        self.n = 0

    def __call__(self, func):
        @wraps(func)
        def wrapped_function(*args, **kwargs):
            log_string = func.__name__ + " was called"
            print('start to timing: ')
            ss = time.time()
            res = func(*args, **kwargs)
            range_time = time.time() - ss
            print('waste time: '+str(range_time))
            return res
        return wrapped_function

# test
@time_count()
def myprint():
    time.sleep(4)
    print(1)
    return 1


if __name__ == "__main__":
    for i in range(1000):
        a = myprint()
        print(a)