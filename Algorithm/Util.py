import matplotlib.pyplot as plt
import numpy as np
import os
class Util:

    @staticmethod
    def savePNG(img, path="./pics"):
        img.save(path, 'PNG')
        return path
