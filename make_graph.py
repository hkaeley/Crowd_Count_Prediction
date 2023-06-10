import matplotlib.pyplot as plt
  
# x = [5,10,15,25,30]
import matplotlib.pyplot as plt
import matplotlib
  
# create data
x = [4,14,24,29]
stockformer = [50.45,70.27,71.84,89.19]
# x2 = [5,15, 25,30]
rnns = [63.51, 60.0 ,57.78,70.56]

matplotlib.rc('xtick', labelsize=8) 
matplotlib.rc('ytick', labelsize=8) 

font = {'family' : 'serif',
        'weight' : 'bold',
        'size'   : 10}

matplotlib.rc('font', **font)
  
# plot lines
plt.plot(x, stockformer, label = "Stockformer", color='cornflowerblue')
plt.plot(x, rnns, label = "Comparison_RNNs", color='indianred')
plt.xlabel("$\it{n}$-day Lag Window", weight='bold')
plt.ylabel("Directional Accuracy", weight='bold')

plt.xticks(ticks=x)
plt.legend()

plt.savefig('accuracy_comp_lighter.PNG')