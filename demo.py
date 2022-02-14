import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
x = np.random.rand(10)
y = pd.date_range(start='2022-01-01', end='2022-01-10')
plt.figure()
plt.plot(x, y, 'r--')
plt.title('x-y')
plt.xtitle('x')
plt.ytitle('y')
plt.show()


