import matplotlib.pyplot as plt
  
# x = [5,10,15,25,30]
import matplotlib.pyplot as plt
import matplotlib


#split into 4 corners - one for each metric
#x-labels = the days
  
# create data
labels_for_each_bar_graph = ["$\it{n}$=4-day", "$\it{n}$=24-day"]
 #[5-day,24-day]
mse_stockformer = [0.005325,0.0862]
mse_lstm = [0.004842,0.04175]
auc_stockformer = [0.9799,0.9727]
auc_lstm = [0.9292,0.9404]
r2_stockformer = [0.9979,0.9124]
r2_lstm = [0.9955,0.9576]
da_stockformer = [0.6036,0.7184]
da_lstm = [0.5676,0.5778]


matplotlib.rc('xtick', labelsize=8) 
matplotlib.rc('ytick', labelsize=8) 

font = {'family' : 'serif',
        'weight' : 'bold',
        'size'   : 8}

matplotlib.rc('font', **font)

fig, axs = plt.subplots(2, 2)

ax_mse = axs[0,0]
ax_auc = axs[0,1]
ax_r2 = axs[1,0]
ax_da = axs[1,1]

X_axis = [1,2]
  
ax_mse.bar([i-0.2 for i in X_axis], mse_stockformer, 0.4, label = 'Stockformer', color = "cornflowerblue")
ax_mse.bar([i+0.2 for i in X_axis], mse_lstm, 0.4, label = 'Baseline LSTM', color = "indianred")
ax_mse.set_ylabel(ylabel='MSE', fontweight='bold')

fontsize = 7.0

for i, v in zip(X_axis, mse_stockformer):
    if  i == 1:
        ax_mse.text(i-0.45, v , str(v), color='black', fontsize = fontsize, fontweight = "bold", fontfamily = "serif")
    else:
        ax_mse.text(i-0.35, v , str(v), color='black', fontsize = fontsize, fontweight = "bold", fontfamily = "serif")

for i, v in zip(X_axis, mse_lstm):
    if i == 0:
        ax_mse.text(i+0.05, v , str(v), color='black', fontsize = fontsize, fontweight = "bold", fontfamily = "serif")
    else:
        ax_mse.text(i+0.03, v , str(v), color='black', fontsize = fontsize, fontweight = "bold", fontfamily = "serif")


ax_auc.bar([i-0.2 for i in X_axis], auc_stockformer, 0.4, label = 'Stockformer', color = "cornflowerblue")
ax_auc.bar([i+0.2 for i in X_axis], auc_lstm, 0.4, label = 'Baseline LSTM', color = "indianred")
ax_auc.set_ylabel(ylabel='AUC', fontweight='bold')

for i, v in zip(X_axis, auc_stockformer):
    ax_auc.text(i-0.35, v , str(v), color='black', fontsize = fontsize, fontweight = "bold", fontfamily = "serif")

for i, v in zip(X_axis, auc_lstm):
    ax_auc.text(i+0.05, v , str(v), color='black', fontsize = fontsize, fontweight = "bold", fontfamily = "serif")


ax_r2.bar([i-0.2 for i in X_axis], r2_stockformer, 0.4, label = 'Stockformer', color = "cornflowerblue")
ax_r2.bar([i+0.2 for i in X_axis], r2_lstm, 0.4, label = 'Baseline LSTM', color = "indianred")
ax_r2.set_ylabel( ylabel='R2', fontweight='bold')

for i, v in zip(X_axis, r2_stockformer):
    ax_r2.text(i-0.35, v , str(v), color='black', fontsize = fontsize, fontweight = "bold", fontfamily = "serif")

for i, v in zip(X_axis, r2_lstm):
    ax_r2.text(i+0.05, v , str(v), color='black', fontsize = fontsize, fontweight = "bold", fontfamily = "serif")


ax_da.bar([i-0.2 for i in X_axis], da_stockformer, 0.4, label = 'Stockformer', color = "cornflowerblue")
ax_da.bar([i+0.2 for i in X_axis], da_lstm, 0.4, label = 'Baseline LSTM', color = "indianred")
ax_da.set_ylabel( ylabel='Directional Accuracy', fontweight='bold')

for i, v in zip(X_axis, da_stockformer):
    ax_da.text(i-0.35, v , str(v), color='black', fontsize = fontsize, fontweight = "bold", fontfamily = "serif")

for i, v in zip(X_axis, da_lstm):
    ax_da.text(i+0.05, v , str(v), color='black', fontsize = fontsize, fontweight = "bold", fontfamily = "serif")


plt.setp((ax_mse, ax_auc, ax_r2, ax_da), xticks=[1, 2], xticklabels=labels_for_each_bar_graph)
  
# plot lines
# plt.xlabel("N-day")
# plt.ylabel("Metric Value")

# plt.legend()

for a in [ax_mse, ax_auc, ax_r2,ax_da]:
    for label in ( a.get_yticklabels()):
        label.set_fontsize(6)

handles, labels = ax_da.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center')



plt.savefig('bar_lighter_smaller_font.PNG')
print("Done")