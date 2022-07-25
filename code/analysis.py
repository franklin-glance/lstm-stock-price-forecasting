import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import dataconfig

pd.options.display.max_columns = None

dc = dataconfig.DataConfig()

dc.getdata(['AAPL'])

df = dc.data['AAPL'].daily

