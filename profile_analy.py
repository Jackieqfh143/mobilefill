import pstats
p=pstats.Stats("profile.stats")
p.sort_stats("cumtime").print_stats(100)
# p.print_callers()  # 可以显示函数被哪些函数调用
# p.print_callees()  # 可以显示哪个函数调用了哪些函数


"""
ncalls：表示函数调用的次数；
tottime：表示指定函数的总的运行时间，除掉函数中调用子函数的运行时间；
percall：（第一个percall）等于 tottime/ncalls；
cumtime：表示该函数及其所有子函数的调用运行的时间，即函数开始调用到返回的时间；
percall：（第二个percall）即函数运行一次的平均时间，等于 cumtime/ncalls；
filename:lineno(function)：每个函数调用的具体信息；
"""