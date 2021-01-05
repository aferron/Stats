# practice problems for stat 551

import stemgraphic
#import matplotlib.pyplot as plt
#fig, ax = plt.plot(range(10))
#fig.savefig("demo.pdf")

data = [56, 74, 25, 68, 67, 68, 90, 78, 85, 83, 46, 93]
fig, ax = stemgraphic.stem_graphic(data, scale = 10)
fig.savefig("demo.pdf")
