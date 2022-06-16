def timeit(func, nr_runs):
    import time

    def wrapper(**args):
        total = 0

        for i in range(nr_runs):
            start = time.process_time_ns()

            func(**args)

            end = time.process_time_ns()
            total += end - start

        print(f"{nr_runs} executed, one run took {float(total) / nr_runs} ns")

    return wrapper
