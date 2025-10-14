import os
import tpysa

if __name__ == "__main__":
    n_time = 24
    n_elements = 2700

    dir_name = os.path.dirname(__file__)
    main_file = os.path.join(dir_name, "main.py")

    iterator = tpysa.FixedStress(main_file, n_elements, n_time)

    iterator.anderson(10)
    # iterator.fixed_point(10)
