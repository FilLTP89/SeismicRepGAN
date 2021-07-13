import numpy as np
import pandas as pd
import seaborn as sn
from scipy import signal
from scipy.stats import norm
import obspy.signal
from obspy.signal.tf_misfit import plot_tf_gofs, eg, pg
import itertools
from matplotlib import pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.sans-serif'] = ['Tahoma']
families = ['serif', 'sans-serif', 'cursive', 'fantasy', 'monospace']
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

def arias_intensity(dtm,tha,pc=0.95,nf=9.81):
    aid = np.pi/2./nf*np.cumtrapz(tha**2, dx=dtm, axis=-1, initial = 0.)
    mai = np.max(aid,axis=-1)
    ait = np.empty_like(mai)
    idx = np.empty_like(mai)
    if mai.size>1:
        for i in range(mai.size):
            ths = np.where(aid[i,...]/mai[i]>=pc)[0][0]
            ait[i] = aid[i,ths]
            idx[i] = ths*dtm
    else:
        ths = np.where(aid/mai>=pc)[0][0]
        ait = aid[ths]
        idx = ths*dtm
    return aid,ait,idx

def PlotLoss(history):
    # Plot loss
    hfg = plt.figure(figsize=(12,6))
    hax = hfg.add_subplot(111)
    #hax.plot(history.history['AdvDloss'], color='b')
    #hax.plot(history.history['AdvGloss'], color='g')
    hax.plot(history.history['AdvDlossX'], color='r')
    hax.plot(history.history['AdvDlossC'], color='c')
    hax.plot(history.history['AdvDlossS'], color='m')
    hax.plot(history.history['AdvDlossN'], color='gold')
    #hax.plot(history.history['AdvDlossPenGradX'])
    hax.plot(history.history['AdvGlossX'], color = 'purple')
    hax.plot(history.history['AdvGlossC'], color = 'brown')
    hax.plot(history.history['AdvGlossS'], color = 'salmon')
    hax.plot(history.history['AdvGlossN'], color = 'lightblue')
    hax.plot(history.history['RecGlossX'], color='darkorange')
    hax.plot(history.history['RecGlossC'], color='lime')
    hax.plot(history.history['RecGlossS'], color='grey')
    hax.set_title('Model loss', fontsize=18)
    hax.set_ylabel('Loss', fontsize=14)
    hax.set_xlabel('Epoch', fontsize=14)
    #hax.legend(['AdvDloss', 'AdvGloss','AdvDlossX','AdvDlossC','AdvDlossS','AdvDlossN','AdvGlossX',
    #    'AdvGlossC','AdvGlossS','AdvGlossN','RecGlossX','RecGlossC','RecGlossS'], loc='lower right')
    hax.legend(['AdvDlossX','AdvDlossC','AdvDlossS','AdvDlossN','AdvGlossX',
        'AdvGlossC','AdvGlossS','AdvGlossN','RecGlossX','RecGlossC','RecGlossS'], loc='lower right')
    plt.tight_layout()
    plt.savefig('loss.png',bbox_inches = 'tight')
    plt.close()

def PlotReconstructedTHs(model,realXC):
    # Plot reconstructed time-histories
    realX, realC = realXC
    # realX = np.concatenate([x for x, c in X], axis=0)
    # realC = np.concatenate([c for x, c in X], axis=0)
    fakeX,fakeC,fakeS,fakeN,fakeX_res = model.predict(X)

    # Print real vs reconstructed signal
    hfg = plt.figure(figsize=(12,6))
    hax = hfg.add_subplot(111)
    hax.plot(realX[0,:,0], color='black')
    hax.plot(fakeX[0,:,0], color='orange')
    hax.set_title('X reconstruction', fontsize=18)
    hax.set_ylabel('X', fontsize=14)
    hax.set_xlabel('t', fontsize=14)
    hax.legend(['X', 'G(F(X))'], loc='lower right')
    plt.tight_layout()
    plt.savefig('reconstruction.png',bbox_inches = 'tight')
    plt.close()

    # Print reconstructed signal after fakeN resampling
    hfg = plt.figure(figsize=(12,6))
    hax = hfg.add_subplot(111)
    hax.plot(realX[0,:,0], color='black')
    hax.plot(fakeX[0,:,0], color='orange')
    hax.plot(fakeX_res[0,:,0], color='green')
    hax.set_title('X reconstruction', fontsize=18)
    hax.set_ylabel('X', fontsize=14)
    hax.set_xlabel('t', fontsize=14)
    hax.legend(['X', 'G(F(X))', 'G(F(X)) res'], loc='lower right')
    plt.tight_layout()
    plt.savefig('resampling.png',bbox_inches = 'tight')
    plt.close()

def PlotCorrelationS(model,realXC):
    # Plot s correlation
    realX, realC = realXC
    # realX = np.concatenate([x for x, c in X], axis=0)
    # realC = np.concatenate([c for x, c in X], axis=0)
    fakeX,fakeC,fakeS,fakeN,fakeX_res = model.predict(X)

    # Print fakeS autocorrelation
    S = np.reshape(fakeS, fakeS.size)
    hfg = plt.figure(figsize=(12,6))
    corr3 = hfg.add_subplot(111)
    corr3.set_title("Continuous variables S - Autocorrelation Plot", fontsize=18)
    corr3.set_xlabel("Lags", fontsize=14)
    corr3.set_ylabel("Autocorrelation", fontsize=14)
    plt.acorr(S, maxlags = 20)
    plt.tight_layout()
    plt.savefig('autocorrelation_fakeS.png',bbox_inches = 'tight')
    plt.close()

    # Print fakeS correlation matrix
    corrMatrix = np.corrcoef(fakeS)
    ax = sn.heatmap(corrMatrix, vmin=-1, vmax=1, center=0,
        cmap=sn.diverging_palette(20, 220, n=200), square=True)
    #ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.set_title("Continuous variables S - Correlation matrix", fontsize=18)
    plt.savefig('correlation_matrix.png',bbox_inches = 'tight')
    plt.close()

    # Print fakeS distribution
    fakeS_std = np.std(fakeS)
    fakeS_mean = np.mean(fakeS)
    num_bins = 50
    n, bins, patches = plt.hist(fakeS, num_bins, density=True, facecolor='blue', alpha=0.5)
    # add a 'best fit' line
    y = ((1 / (np.sqrt(2 * np.pi) * fakeS_std)) *np.exp(-0.5 * (1 / fakeS_std * (bins - fakeS_mean))**2))
    #y = norm.pdf(bins, fakeS_mean, fakeS_std)
    plt.plot(bins, y, 'r--')
    plt.xlabel('fake S')
    plt.ylabel('Probability density')
    plt.title('Continuous variables S - Distribution')
    plt.savefig('fakeS_distribution.png',bbox_inches = 'tight')
    plt.close()

def PlotDistributionN(model,realXC):
    # Plot n distribution
    realX, realC = realXC
    # realX = np.concatenate([x for x, c in X], axis=0)
    # realC = np.concatenate([c for x, c in X], axis=0)
    fakeX,fakeC,fakeS,fakeN,fakeX_res = model.predict(X)
    # Print fakeN distribution
    fakeN_std = np.std(fakeN)
    fakeN_mean = np.mean(fakeN)
    num_bins = 50
    n, bins, patches = plt.hist(fakeN, num_bins, density=True, facecolor='blue', alpha=0.5)
    # add a 'best fit' line
    y = ((1 / (np.sqrt(2 * np.pi) * fakeN_std)) *np.exp(-0.5 * (1 / fakeN_std * (bins - fakeN_mean))**2))
    #y = norm.pdf(bins, fakeN_mean, fakeN_std)
    plt.plot(bins, y, 'r--')
    plt.xlabel('fake N')
    plt.ylabel('Probability density')
    plt.title('Noise N - Distribution')
    plt.savefig('fakeN_distribution.png',bbox_inches = 'tight')
    plt.close()
    # expl_var = explained_variance_score(realC,fakeC, multioutput='raw_values')
    # mae = mean_absolute_error(realC,fakeC, multioutput='raw_values')
    # mse = mean_squared_error(realC,fakeC,multioutput='raw_values')
    # msle = mean_squared_log_error(realC,fakeC,multioutput='raw_values')

    # file = open("metricsC.txt", "w")
    # file.write("\n explained variance score: %s \n" % expl_var)
    # file.write("\n mean absolute error: %s \n" % mae)
    # file.write("\n mean squared error: %s \n" % mse)
    # file.write("\n mean squared log error: %s \n" % msle)
    # file.close()


def PlotTHSGoFs(model,realXC):
    # Plot reconstructed time-histories
    realX, realC = realXC
    # realX = np.concatenate([x for x, c in X], axis=0)
    # realC = np.concatenate([c for x, c in X], axis=0)
    fakeX,fakeC,fakeS,fakeN,fakeX_res = model.predict(X)

    ## Print signal GoF
    for (i,j) in range((realX.shape[0]),(realX.shape[0])):
        a,b = realX[i, :, 0],fakeX[j,:,0]
        #a,b = realX[i, 1, :],fakeX[j,1,:]
        (swa,swb) = (np.argmax(a),np.argmax(b))
        _,_,swa = arias_intensity(0.001,a,0.05)
        _,_,swb = arias_intensity(0.001,b,0.05)
        if swb>swa:
            pads=(swb-swa,0)
            a=np.pad(a[:-pads[0]],pads,'constant',constant_values=(0,0))
        elif swb<swa:
            pads=(0,swa-swb)
            a=np.pad(a[pads[1]:],pads,'constant',constant_values=(0,0))

    plot_tf_gofs(a,b,dt=0.001,t0=0.0,nf=100,w0=6,norm='global',st2_isref=True,
        a=10.,k=1.,left=0.1,bottom=0.125, h_1=0.2,h_2=0.125,h_3=0.2,
        w_1=0.2,w_2=0.6,w_cb=0.01, d_cb=0.0,show=False,
        plot_args=['k', 'r', 'b'],ylim=0., clim=0.)
    plt.savefig("gof_{:>d}.png".format(i),bbox_inches = 'tight')
    plt.close()

    plot_tf_gofs(realX[0,:,0],fakeX[0,:,0],dt=0.001,t0=0.0,nf=100,w0=6,norm='global',st2_isref=True,
        a=10.,k=1.,left=0.1,bottom=0.125, h_1=0.2,h_2=0.125,h_3=0.2,
        w_1=0.2,w_2=0.6,w_cb=0.01, d_cb=0.0,show=False,
        plot_args=['k', 'r', 'b'],ylim=0., clim=0.)
    plt.savefig("gofs.png",bbox_inches = 'tight')
    plt.close()

def colored_scatter(*args, **kwargs):
    plt.scatter(*args, **kwargs)
    return

def PlotEGPGgrid(col_x, col_y, col_k, df, k_is_color=False, scatter_alpha=.7):
    k=0
    for name, df_group in df.groupby(col_k):
        k+=1
    plt.figure(figsize=(10,6), dpi= 500)
    sn.set_palette("bright")
    g = sn.JointGrid(x=col_x,y=col_y,data=df)
    color = None
    legends=[]
    for name, df_group in df.groupby(col_k):
        legends.append(name)
        if k_is_color:
            color=name
        g.plot_joint(colored_scatter(df_group[col_x],df_group[col_y],color),)
        hax=sn.distplot(df_group[col_x].values,ax=g.ax_marg_x,kde=False,
            color=color,norm_hist=True)
        hay=sn.distplot(df_group[col_y].values,ax=g.ax_marg_y,kde=False,
            color=color,norm_hist=True,vertical=True)
        hax.set_xticks(list(np.linspace(0,10,11)))
        hay.set_yticks(list(np.linspace(0,10,11)))
    ## Do also global Hist:
    g.ax_joint.set_xticks(list(np.linspace(0,10,11)))
    g.ax_joint.set_yticks(list(np.linspace(0,10,11)))
    plt.legend(legends)
    plt.xlabel(r'$EG$')
    plt.ylabel(r'$PG$')
    plt.savefig('Gz(Fx(X))_gofs.png',bbox_inches = 'tight')
    plt.close()
    return 

def PlotBatchGoFs(model,realXC):
    # Plot GoFs on a batch
    realX, realC = realXC
    # realX = np.concatenate([x for x, c in X], axis=0)
    # realC = np.concatenate([c for x, c in X], axis=0)
    fakeX,fakeC,fakeS,fakeN,fakeX_res = model.predict(X)

    egpg = []

    for b,_ in enumerate(realX):
        for (i,j) in zip(range(realX.shape[0],fakeX.shape[0])):
            #st1 = fakeX[j,:,:].squeeze()
            st1 = tf.keras.backend.squeeze(fakeX[j,:,:])
            #st2 = realX[i,:,:].squeeze()
            st2 = tf.keras.backend.squeeze(realX[i,:,:])
            egpg[b*realX.shape[0]+i,0] = eg(st1,st2,dt=0.001,norm='global',st2_isref=True,a=10.,k=1.).mean()
            egpg[b*realX.shape[0]+i,1] = pg(st1,st2,dt=0.001,norm='global',st2_isref=True,a=10.,k=1.).mean()

    egpg_df = pd.DataFrame(egpg,columns=['EG','PG'])
    egpg_df['kind']=r"$G_z(F_x(X))$"
    PlotEGPGgrid('EG','PG','kind',df=egpg_df)

def PlotClassificationMetrics(model,realXC):
    # Plot classification metrics
    realX, realC = realXC
    # realX = np.concatenate([x for x, c in X], axis=0)
    # realC = np.concatenate([c for x, c in X], axis=0)
    fakeX,fakeC,fakeS,fakeN,fakeX_res = model.predict(X)

    fakeC = fakeC.astype(int)    
    target_names = ['class 0', 'class 1', 'class 2', 'class 3', 'class 4'] 
    report = classification_report(y_true = np.argmax(realC, axis=1), y_pred = np.argmax(fakeC, axis=1),
        target_names=target_names,output_dict=True)
    df = pd.DataFrame(report).transpose()
    df.to_csv('Classification Report C.csv', index= True)
    cr = sn.heatmap(pd.DataFrame(report).iloc[:-1, :].T, annot=True)
    cr.set_title("Categorical variables C - Classification report", fontsize=18)
    plt.savefig('classification_report.png',bbox_inches = 'tight')
    plt.close()
    return