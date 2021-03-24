import concurrent.futures
import time

from typing import Tuple

# Executes a process that takes 4 seconds and a process that takes 10 seconds
# at the same time.
#    result1: "This process took 4 seconds"
#    result2: "This process took 10 seconds"
# time_taken: 10.001785 seconds

def concurrency_demo(sleep_time1: int, sleep_time2: int, executor: concurrent.futures.Executor) -> Tuple[str, str]:
    def long_process(sleep_time: int) -> str:
        time.sleep(sleep_time)
        return 'This process took %r seconds' % sleep_time
    future1 = executor.submit(long_process, sleep_time1)
    future2 = executor.submit(long_process, sleep_time2)
    return future1.result(), future2.result()

sleep_time1, sleep_time2 = 4, 10
executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

start_time = time.time()
result1, result2 = concurrency_demo(sleep_time1, sleep_time2, executor)
end_time = time.time()

print('   result1: "%s"' % result1)
print('   result2: "%s"' % result2)
print('time_taken: %f seconds' % (end_time - start_time))