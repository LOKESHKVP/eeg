import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import scipy.signal as ss
from scipy import signal
import streamlit as st
st.markdown("<h1 style ='color:black; text_align:center;font-family:times new roman;font-size:20pt; font-weight: bold;'>Analysis of eeg signal</h1>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align:center; color:white;background-color:black;font-size:14pt'>📂 Upload your CSV or Excel file. (200MB max) 📂</h1>", unsafe_allow_html=True)
uploaded_file = st.file_uploader(label="",type=['txt'])


global df
if uploaded_file is not None:
   print(uploaded_file)
   
   try:
      df = pd.read_csv(uploaded_file,skiprows=6,header=None)
      st.write(df)
   except:
      st.write("file not found")
      
#df = pd.read_csv("priyadharshini_1.txt",skiprows=6,header=None)
#df = pd.read_csv(uploaded_file,skiprows=6,header=None)
df.columns=['index','channel1','channel2','channel3','channel4','channel5','channel6','channel7','channel8','acc1','acc2','acc3','time_std','timestamp']
df.drop(['index'],axis=1,inplace=True)
st.write(df)
df['channel1'] = ss.detrend(df['channel1'])
df['channel2'] = ss.detrend(df['channel2'])
df['channel3'] = ss.detrend(df['channel3'])
df['channel4'] = ss.detrend(df['channel4'])
df['channel5'] = ss.detrend(df['channel5'])
df['channel6'] = ss.detrend(df['channel6'])
df['channel7'] = ss.detrend(df['channel7'])
df['channel8'] = ss.detrend(df['channel8'])
st.write(df)

chan1 = (df['channel1']-np.mean(df['channel1']))/np.std(df['channel1'])
chan2 = (df['channel2']-np.mean(df['channel2']))/np.std(df['channel2'])
chan3 = (df['channel3']-np.mean(df['channel3']))/np.std(df['channel3'])
chan4 = (df['channel4']-np.mean(df['channel4']))/np.std(df['channel4'])
chan5 = (df['channel5']-np.mean(df['channel5']))/np.std(df['channel5'])
chan6 = (df['channel6']-np.mean(df['channel6']))/np.std(df['channel6'])
chan7 = (df['channel7']-np.mean(df['channel7']))/np.std(df['channel7'])
chan8 = (df['channel8']-np.mean(df['channel8']))/np.std(df['channel8'])

# df['channel1'] = (df['channel1']-np.mean(df['channel1']))/np.std(df['channel1'])
# df['channel2'] = (df['channel2']-np.mean(df['channel2']))/np.std(df['channel2'])
# df['channel3'] = (df['channel3']-np.mean(df['channel3']))/np.std(df['channel3'])
# df['channel4'] = (df['channel4']-np.mean(df['channel4']))/np.std(df['channel4'])
# df['channel5'] = (df['channel5']-np.mean(df['channel5']))/np.std(df['channel5'])
# df['channel6'] = (df['channel6']-np.mean(df['channel6']))/np.std(df['channel6'])
# df['channel7'] = (df['channel7']-np.mean(df['channel7']))/np.std(df['channel7'])
# df['channel8'] = (df['channel8']-np.mean(df['channel8']))/np.std(df['channel8'])
st.write(df)
for column in [chan1, chan2,chan3, chan4,chan5, chan6,chan7, chan8]:    
    plt.plot(column,label="channel"+str(i))
    plt.legend(loc='best')
    st.write(plt)
    i+=1

b, a = ss.iirfilter(1, Wn=50, fs=250, btype="high", ftype="butter")
print(b, a, sep="\n")
# df['channel1'] = ss.filtfilt(b, a, df['channel1'])
# df['channel2'] = ss.filtfilt(b, a, df['channel2'])
# df['channel3'] = ss.filtfilt(b, a, df['channel3'])
# df['channel4'] = ss.filtfilt(b, a, df['channel4'])
# df['channel5'] = ss.filtfilt(b, a, df['channel5'])
# df['channel6'] = ss.filtfilt(b, a, df['channel6'])
# df['channel7'] = ss.filtfilt(b, a, df['channel7'])
# df['channel8'] = ss.filtfilt(b, a, df['channel8'])

chan1 = ss.filtfilt(b, a, chan1)
chan2 = ss.filtfilt(b, a, chan2)
chan3 = ss.filtfilt(b, a, chan3)
chan4 = ss.filtfilt(b, a, chan4)
chan5 = ss.filtfilt(b, a, chan5)
chan6 = ss.filtfilt(b, a, chan6)
chan7 = ss.filtfilt(b, a, chan7)
chan8 = ss.filtfilt(b, a, chan8)
st.write(df)
i=1
for column in [zchan1, zchan2,zchan3, zchan4,zchan5, zchan6,zchan7, zchan8]:    
    plt.plot(column,label="channel"+str(i))
    plt.legend(loc='best')
    st.write(plt)
    i+=1
      
chan1[np.abs(chan1)>3] = 0
chan2[np.abs(chan2)>3] = 0
chan3[np.abs(chan3)>3] = 0
chan4[np.abs(chan4)>3] = 0
chan5[np.abs(chan5)>3] = 0
chan6[np.abs(chan6)>3] = 0
chan7[np.abs(chan7)>3] = 0
chan8[np.abs(chan8)>3] = 0
i=1
for column in [chan1, chan2,chan3, chan4,chan5, chan6,chan7, chan8]:    
    plt.plot(column,label="channel"+str(i))
    plt.legend(loc='best')
    st.write(plt)
    i+=1

df['channel1'] = chan1
df['channel2'] = chan2
df['channel3'] = chan3
df['channel4'] = chan4
df['channel5'] = chan5
df['channel6'] = chan6
df['channel7'] = chan7
df['channel8'] = chan8

for column in df[['channel1', 'channel2','channel3', 'channel4','channel5', 'channel6','channel7', 'channel8']]:    
    plt.plot(df[column],label=column)
    plt.legend(loc='best')
    st.write(plt)
   
from scipy.integrate import simps
import scipy.stats as sst
from matplotlib.mlab import psd

def bandpower(trace,band):
    [a1,f1]=psd(trace[~np.isnan(trace)],512,Fs=250)
    #print(a1,f1)
    total_power1 = simps(a1, dx=0.1)
    #print(total_power1)
    ap1 = simps(a1[(f1>band[0]) & (f1<band[1])], dx=0.1)
    return ap1/total_power1
    
alpha = np.zeros((600,8))
beta = np.zeros((600,8))
gamma = np.zeros((600,8))
theta = np.zeros((600,8))

c=0
for i in np.arange(0,61301,250):
    
    X1=df['channel1']
    X2=df['channel2']
    X3=df['channel3']
    X4=df['channel4']
    X5=df['channel5']    
    X6=df['channel6']
    X7=df['channel7']
    X8=df['channel8']
    
    X1=X1[i:i+250]
    X2=X2[i:i+250]
    X3=X3[i:i+250]
    X4=X4[i:i+250]
    X5=X5[i:i+250]
    X6=X6[i:i+250]
    X7=X7[i:i+250]  
    X8=X8[i:i+250] 
    
    alpha[c,0] = bandpower(X1,[8,12])
    alpha[c,1] = bandpower(X2,[8,12])
    alpha[c,2] = bandpower(X3,[8,12])
    alpha[c,3] = bandpower(X4,[8,12])
    alpha[c,4] = bandpower(X5,[8,12])
    alpha[c,5] = bandpower(X6,[8,12])
    alpha[c,6] = bandpower(X7,[8,12])
    alpha[c,7] = bandpower(X8,[8,12])
    
    beta[c,0] = bandpower(X1,[12,30])
    beta[c,1] = bandpower(X2,[12,30])
    beta[c,2] = bandpower(X3,[12,30])
    beta[c,3] = bandpower(X4,[12,30])
    beta[c,4] = bandpower(X5,[12,30])
    beta[c,5] = bandpower(X6,[12,30])
    beta[c,6] = bandpower(X7,[12,30])
    beta[c,7] = bandpower(X8,[12,30])

        
    gamma[c,0] = bandpower(X1,[30,100])
    gamma[c,1] = bandpower(X2,[30,100])
    gamma[c,2] = bandpower(X3,[30,100])
    gamma[c,3] = bandpower(X4,[30,100])
    gamma[c,4] = bandpower(X5,[30,100])
    gamma[c,5] = bandpower(X6,[30,100])
    gamma[c,6] = bandpower(X7,[30,100])
    gamma[c,7] = bandpower(X8,[30,100])
    
        
    theta[c,0] = bandpower(X1,[4,7])
    theta[c,1] = bandpower(X2,[4,7])
    theta[c,2] = bandpower(X3,[4,7])
    theta[c,3] = bandpower(X4,[4,7])
    theta[c,4] = bandpower(X5,[4,7])
    theta[c,5] = bandpower(X6,[4,7])
    theta[c,6] = bandpower(X7,[4,7])
    theta[c,7] = bandpower(X8,[4,7])
    
    
    c+=1
    
alpha_bands = pd.DataFrame(alpha, columns = ['alpha_power_1','alpha_power_2','alpha_power_3','alpha_power_4','alpha_power_5','alpha_power_6','alpha_power_7','alpha_power_8'])
beta_bands = pd.DataFrame(beta, columns = ['beta_power_1','beta_power_2','beta_power_3','beta_power_4','beta_power_5','beta_power_6','beta_power_7','beta_power_8'])
gamma_bands = pd.DataFrame(gamma, columns = ['gamma_power_1','gamma_power_2','gamma_power_3','gamma_power_4','gamma_power_5','gamma_power_6','gamma_power_7','gamma_power_8'])
theta_bands = pd.DataFrame(theta, columns = ['theta_power_1','theta_power_2','theta_power_3','theta_power_4','theta_power_5','theta_power_6','theta_power_7','theta_power_8'])
#st.write(beta_bands)

channels=['FP1','FP2','C3','C4','T5','T6','O1','O2']

no=len(df)
f, psd = ss.welch(df['channel1'], fs=250,nperseg=61302)
beta_power1 = np.sum(psd[(f >= 13) & (f <= 22)]) # Compute the beta wave power

f, psd = ss.welch(df['channel2'], fs=250,nperseg=61302)
beta_power2 = np.sum(psd[(f >= 13) & (f <= 22)]) 

f, psd = ss.welch(df['channel3'], fs=250,nperseg=61302)
beta_power3 = np.sum(psd[(f >= 13) & (f <= 22)]) 

f, psd = ss.welch(df['channel4'], fs=250,nperseg=61302)
beta_power4 = np.sum(psd[(f >= 13) & (f <= 22)]) 

f, psd = ss.welch(df['channel5'], fs=250,nperseg=61302)
beta_power5 = np.sum(psd[(f >= 13) & (f <= 22)]) 

f, psd = ss.welch(df['channel6'], fs=250,nperseg=61302)
beta_power6 = np.sum(psd[(f >= 13) & (f <= 22)]) 

f, psd = ss.welch(df['channel7'], fs=250,nperseg=61302)
beta_power7 =np.sum(psd[(f >= 13) & (f <= 22)]) 

f, psd = ss.welch(df['channel8'], fs=250,nperseg=61302)
beta_power8 = np.sum(psd[(f >= 13) & (f <= 22)])   

beta_power=[beta_power1,beta_power2,beta_power3,beta_power4,beta_power5,beta_power6,beta_power7,beta_power8]

print(beta_power1,beta_power2,beta_power3,beta_power4,beta_power5,beta_power6,beta_power7,beta_power8)

import plotly.express as px

fig = px.bar(df, x=channels, y=beta_power, color=channels,
              pattern_shape_sequence=[".", "x", "+"])
fig.show()
st.info("BETA waves")
#with st.beta_expander("Write a review 📝"):
st.write(fig)
plt.figure(figsize=[6.4, 2.4])  
for column in beta_bands[['beta_power_1','beta_power_2','beta_power_3','beta_power_4','beta_power_5','beta_power_6','beta_power_7','beta_power_8']]:    
    plt.plot(beta_bands[column][0:100],label=column)
    plt.xlabel("Time / s")
    plt.ylabel("beta power")
    plt.legend(loc="lower center", bbox_to_anchor=[0.5, 1],ncol=2, fontsize="smaller")
st.pyplot(plt)


no=len(df)
f, psd = ss.welch(df['channel1'], fs=250,nperseg=61302)
alpha_power1 = np.sum(psd[(f >= 8) & (f <= 13)]) # Compute the alpha wave power

f, psd = ss.welch(df['channel2'], fs=250,nperseg=61302)
alpha_power2 = np.sum(psd[(f >= 8) & (f <= 13)]) 

f, psd = ss.welch(df['channel3'], fs=250,nperseg=61302)
alpha_power3 = np.sum(psd[(f >= 8) & (f <= 13)]) 

f, psd = ss.welch(df['channel4'], fs=250,nperseg=61302)
alpha_power4 = np.sum(psd[(f >= 8) & (f <= 13)]) 

f, psd = ss.welch(df['channel5'], fs=250,nperseg=61302)
alpha_power5 = np.sum(psd[(f >= 8) & (f <= 13)]) 

f, psd = ss.welch(df['channel6'], fs=250,nperseg=61302)
alpha_power6 = np.sum(psd[(f >= 8) & (f <= 13)]) 

f, psd = ss.welch(df['channel7'], fs=250,nperseg=61302)
alpha_power7 = np.sum(psd[(f >= 8) & (f <= 13)]) 

f, psd = ss.welch(df['channel8'], fs=250,nperseg=61302)
alpha_power8 = np.sum(psd[(f >= 8) & (f <= 13)])  

alpha_power=[alpha_power1,alpha_power2,alpha_power3,alpha_power4,alpha_power5,alpha_power6,alpha_power7,alpha_power8]

print(alpha_power1,alpha_power2,alpha_power3,alpha_power4,alpha_power5,alpha_power6,alpha_power7,alpha_power8)

import plotly.express as px

fig = px.bar(df, x=channels, y=alpha_power, color=channels,
              pattern_shape_sequence=[".", "x", "+"])
fig.show()
st.info("Alpha waves")
st.write(fig)
plt.figure(figsize=[6.4, 2.4])  
for column in alpha_bands[['alpha_power_1','alpha_power_2','alpha_power_3','alpha_power_4','alpha_power_5','alpha_power_6','alpha_power_7','alpha_power_8']]:    
    plt.plot(alpha_bands[column][0:100],label=column)
    plt.xlabel("Time / s")
    plt.ylabel("alpha power")
    plt.legend(loc="lower center", bbox_to_anchor=[0.5, 1],ncol=2, fontsize="smaller")
st.pyplot(plt)


no=len(df)
f, psd = ss.welch(df['channel1'], fs=250,nperseg=61302)
gamma_power1 = np.sum(psd[(f >= 30) & (f <= 100)])  # Compute the alpha wave power

f, psd = ss.welch(df['channel2'], fs=250,nperseg=61302)
gamma_power2 = np.sum(psd[(f >= 30 )& (f <= 100)]) 

f, psd = ss.welch(df['channel3'], fs=250,nperseg=61302)
gamma_power3 = np.sum(psd[(f >= 30) & (f <= 100)]) 

f, psd = ss.welch(df['channel4'], fs=250,nperseg=61302)
gamma_power4 = np.sum(psd[(f >= 30) & (f <= 100)]) 

f, psd = ss.welch(df['channel5'], fs=250,nperseg=61302)
gamma_power5 = np.sum(psd[(f >= 30) & (f <= 100)]) 

f, psd = ss.welch(df['channel6'], fs=250,nperseg=61302)
gamma_power6 = np.sum(psd[(f >= 30) & (f <= 100)]) 

f, psd = ss.welch(df['channel7'], fs=250,nperseg=61302)
gamma_power7 = np.sum(psd[(f >= 30) & (f <= 100)]) 

f, psd = ss.welch(df['channel8'], fs=250,nperseg=61302)
gamma_power8 = np.sum(psd[(f >= 30) & (f <= 100)])  

gamma_power=[gamma_power1,gamma_power2,gamma_power3,gamma_power4,gamma_power5,gamma_power6,gamma_power7,gamma_power8]


import plotly.express as px

fig = px.bar(df, x=channels, y=gamma_power, color=channels,
              pattern_shape_sequence=[".", "x", "+"])
fig.show()
st.info("Gamma waves")
st.write(fig)
plt.figure(figsize=[6.4, 2.4])  
for column in gamma_bands[['gamma_power_1','gamma_power_2','gamma_power_3','gamma_power_4','gamma_power_5','gamma_power_6','gamma_power_7','gamma_power_8']]:    
    plt.plot(gamma_bands[column][0:100],label=column)
    plt.xlabel("Time / s")
    plt.ylabel("gamma power")
    plt.legend(loc="lower center", bbox_to_anchor=[0.5, 1],ncol=2, fontsize="smaller")
st.pyplot(plt)

no=len(df)
f, psd = ss.welch(df['channel1'], fs=250,nperseg=61302)
delta_power1 = np.sum(psd[(f >= 0.5) & (f <= 4)]) 

f, psd = ss.welch(df['channel2'], fs=250,nperseg=61302)
delta_power2 = np.sum(psd[(f >= 0.5) & (f <= 4)])

f, psd = ss.welch(df['channel3'], fs=250,nperseg=61302)
delta_power3 =np.sum(psd[(f >= 0.5) & (f <= 4)]) 

f, psd = ss.welch(df['channel4'], fs=250,nperseg=61302)
delta_power4 = np.sum(psd[(f >= 0.5) & (f <= 4)])

f, psd = ss.welch(df['channel5'], fs=250,nperseg=61302)
delta_power5 = np.sum(psd[(f >= 0.5) & (f <= 4)])

f, psd = ss.welch(df['channel6'], fs=250,nperseg=61302)
delta_power6 =np.sum(psd[(f >= 0.5) & (f <= 4)])

f, psd = ss.welch(df['channel7'], fs=250,nperseg=61302)
delta_power7 =np.sum(psd[(f >= 0.5) & (f <= 4)]) 

f, psd = ss.welch(df['channel8'], fs=250,nperseg=61302)
delta_power8 = np.sum(psd[(f >= 0.5) & (f <= 4)])   

delta_power=[delta_power1,delta_power2,delta_power3,delta_power4,delta_power5,delta_power6,delta_power7,delta_power8]

print(delta_power1,delta_power2,delta_power3,delta_power4,delta_power5,delta_power6,delta_power7,delta_power8)

no=len(df)
f, psd = ss.welch(df['channel1'], fs=250,nperseg=61302)
theta_power1 = np.sum(psd[(f >= 4) & (f <= 7)]) 

f, psd = ss.welch(df['channel2'], fs=250,nperseg=61302)
theta_power2 = np.sum(psd[(f >= 4) & (f <= 7)]) 

f, psd = ss.welch(df['channel3'], fs=250,nperseg=61302)
theta_power3 = np.sum(psd[(f >= 4) & (f <= 7)]) 

f, psd = ss.welch(df['channel4'], fs=250,nperseg=61302)
theta_power4 = np.sum(psd[(f >= 4) & (f <= 7)]) 

f, psd = ss.welch(df['channel5'], fs=250,nperseg=61302)
theta_power5 = np.sum(psd[(f >= 4) & (f <= 7)]) 

f, psd = ss.welch(df['channel6'], fs=250,nperseg=61302)
theta_power6 = np.sum(psd[(f >= 4) & (f <= 7)]) 

f, psd = ss.welch(df['channel7'], fs=250,nperseg=61302)
theta_power7 = np.sum(psd[(f >= 4) & (f <= 7)]) 

f, psd = ss.welch(df['channel8'], fs=250,nperseg=61302)
theta_power8 = np.sum(psd[(f >= 4) & (f <= 7)])   

theta_power=[theta_power1,theta_power2,theta_power3,theta_power4,theta_power5,theta_power6,theta_power7,theta_power8]


      
import plotly.express as px

fig = px.bar(df, x=channels, y=theta_power, color=channels,
              pattern_shape_sequence=[".", "x", "+"])
fig.show()

st.info("Theta waves")
st.write(fig)
plt.figure(figsize=[6.4, 2.4])  
for column in theta_bands[['theta_power_1','theta_power_2','theta_power_3','theta_power_4','theta_power_5','theta_power_6','theta_power_7','theta_power_8']]:    
    plt.plot(theta_bands[column][0:100],label=column)
    plt.xlabel("Time / s")
    plt.ylabel("theta power")
    plt.legend(loc="lower center", bbox_to_anchor=[0.5, 1],ncol=2, fontsize="smaller")
st.pyplot(plt)   

      
left_alpha=np.sum([alpha_power1,alpha_power3,alpha_power5,alpha_power7])
right_alpha=np.sum([alpha_power2,alpha_power4,alpha_power6,alpha_power8])

left_beta=np.sum([beta_power1,beta_power3,beta_power5,beta_power7])
right_beta=np.sum([beta_power2,beta_power4,beta_power6,beta_power8])

left_delta=np.sum([delta_power1,delta_power3,delta_power5,delta_power7])
right_delta=np.sum([delta_power2,delta_power4,delta_power6,delta_power8])

left_gamma=np.sum([gamma_power1,gamma_power3,gamma_power5,gamma_power7])
right_gamma=np.sum([gamma_power2,gamma_power4,gamma_power6,gamma_power8])

left_theta=np.sum([theta_power1,theta_power3,theta_power5,theta_power7])
right_theta=np.sum([theta_power2,theta_power4,theta_power6,theta_power8])




import plotly.graph_objects as px

x = ['left', 'right']
 
plot = px.Figure(data=[px.Bar(
    name = 'alpha (Relax)',
    x = x,
    y = [left_alpha,right_alpha]
   ),
    px.Bar(
    name = 'beta (Engaged)',
    x = x,
    y = [left_beta,right_beta]
   ),
        px.Bar(
    name = 'delta (Deep sleep)',
    x = x,
    y = [left_delta,right_delta]
   ),                   
         px.Bar(
    name = 'gamma (Concentration)',
    x = x,
    y = [left_gamma,right_gamma]
   ),  
          px.Bar(
    name = 'theta (Dowsy)',
    x = x,
    y = [left_theta,right_theta]
   ),                       
])
                  
plot.show()

st.write(plot)


alpha_totalpower=np.sum([alpha_power1,alpha_power3,alpha_power5,alpha_power7,alpha_power2,alpha_power4,alpha_power6,alpha_power8])
beta_totalpower=np.sum([beta_power1,beta_power3,beta_power5,beta_power7,beta_power2,beta_power4,beta_power6,beta_power8])
gamma_totalpower=np.sum([gamma_power1,gamma_power3,gamma_power5,gamma_power7,gamma_power2,gamma_power4,gamma_power6,gamma_power8])
delta_totalpower=np.sum([delta_power1,delta_power3,delta_power5,delta_power7,delta_power2,delta_power4,delta_power6,delta_power8])
theta_totalpower=np.sum([theta_power1,theta_power3,theta_power5,theta_power7,theta_power2,theta_power4,theta_power6,theta_power8])

import plotly.graph_objects as px

bands=['alpha (Relax)','beta (Engaged)','delta (Deep sleep)','gamma (Concentration)','theta (Dowsy)']
powers=[alpha_totalpower,beta_totalpower,gamma_totalpower,delta_totalpower,theta_totalpower]
import plotly.express as px

fig = px.bar(df, x=bands, y=powers, color=bands,
              pattern_shape_sequence=[".", "x", "+"])
fig.show()
                  

st.write(fig)
