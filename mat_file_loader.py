# library imports
import os

# project imports


class MatFileLoader:
    """
    This class responsible to read the .mat file the processes produces
    """

    def __init__(self):
        pass

    @staticmethod
    def read(path: str,
             delete_in_end: bool = False):
        ans = {}
        key = None
        val = []
        with open(path, "r") as data_file:
            for line in data_file.readlines():
                # if a name row, start new var
                if line.startswith("# name: "):
                    # make sure this is not the first one
                    if key is not None:
                        ans[key] = val
                        val = []  # reset the val
                    key = line[len("# name: "):].strip()
                elif line.startswith("#") or len(line.strip()) == 0:  # this is comment line or empty one, skip it
                    continue
                else:
                    # process data
                    items = [float(val) for val in line.strip().split(" ")]
                    # just to make sure we do not have a list of list of size 1
                    if len(items) == 1:
                        val.append(items[0])
                    else:
                        val.append(items)
            if key is not None:
                ans[key] = val
        # mostly for QA, just to make sure we do not read the same file twice
        if delete_in_end:
            os.remove(path)
        return ans
