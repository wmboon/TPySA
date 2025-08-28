import os
import tpysa


if __name__ == "__main__":
    n_time = 40
    n_elements = 10**3

    dir_name = os.path.dirname(__file__)
    main_file = os.path.join(dir_name, "main.py")

    iterator = tpysa.FixedPoint(main_file, n_elements, n_time)

    # iterator.anderson(9)
    iterator.fixed_point(4)
