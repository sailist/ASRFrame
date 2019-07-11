from util.reader import Thchs30
import os

dir_path = os.path.split(os.path.realpath(__file__))[0]
py_file = os.path.join(dir_path,"thupy.txt")

def create_dict_by_thchs30(path):
    thu = Thchs30(path)
    _,y_set = thu.load_from_path()
    y_set = [os.path.splitext(i)[0]+".trn" for i in y_set]

    pyset = set()
    for y in y_set:
        with open(y,encoding="utf-8") as f:
            f.readline()
            line = f.readline().strip()
            pylist = line.split(" ")
            pyset.update(pylist)

        with open(py_file,"w",encoding="utf-8") as w:
            for i in pyset:
                w.write(f"{i}\n")
