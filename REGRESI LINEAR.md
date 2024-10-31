import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

df=pd.DataFrame([[151,63],[174,81],[138,56],[186,91],
                 [128,47],[136,57],[179,76],[163,72],[152,62],[131,48]])
df.columns=['tinggi','berat']
df

df.columns=['x','y']
print(df)

df.corr()
    x_train=df['x'].values[:,np.newaxis]
y_train=df['y'].values

lm=LinearRegression()
lm.fit(x_train,y_train)

LinearRegression()
x_test=[[170],[171],[160],[180],[150]]
p=lm.predict(x_test)
print(p)

print('Coefficient:'+str(lm.coef_))
print('Intercept:'+str(lm.intercept_))

pb=lm.predict(x_train)
dfc=pd.DataFrame({'x':df['x'],'y':pb})
plt.scatter(df['x'],df['y'])
plt.plot(dfc['x'],dfc['y'],color='red',linewidth=1)
plt.xlabel('Tinggi dalam cm')
plt.ylabel('Berat dalam kg')
plt.show

y_asli=[72,62,48]
y_hasil_prediksi=lm.predict([[163],[152],[131]])
print(y_hasil_prediksi)

from sklearn.metrics import r2_score
akurasi =r2_score(y_asli,y_hasil_prediksi)
print(akurasi*100)

