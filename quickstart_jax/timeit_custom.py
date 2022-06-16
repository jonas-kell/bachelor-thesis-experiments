def timeit(func, nr_runs, **args):
    import time

    def wrapper():
        total = 0

        for i in range(nr_runs):
            start = time.process_time_ns()

            func(**args)

            end = time.process_time_ns()
            total += end - start

        print(f"{nr_runs} executed, one runtook {float(total) / nr_runs} ns")

    wrapper()
