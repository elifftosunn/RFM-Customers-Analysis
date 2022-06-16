import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from mlxtend.frequent_patterns import fpgrowth 


df = pd.read_csv("OnlineRetail.csv",encoding= 'unicode_escape')
# to convert InvoiceDate from string to datetime
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
# adding to a TotalPrice column
df["TotalPrice"] = df.Quantity * df.UnitPrice
# positive Quantity and TotalPrice
df = df[df["Quantity"] > 0]
df = df[df["TotalPrice"] > 0]
# Recency, Monetary and Frequency Values
today = df["InvoiceDate"].max()
firstData = df.groupby("CustomerID").agg({"TotalPrice": lambda x:x.sum(),#monetary
                                               "InvoiceDate": lambda x:(today-x.max()).days}) # recency
secondData = df.groupby(["CustomerID","InvoiceNo"]).agg({"TotalPrice": lambda x:x.sum()})

thirdData = secondData.groupby("CustomerID").agg({"TotalPrice":lambda x:len(x)}) # frequency

rfm_table = pd.merge(firstData,thirdData, on = "CustomerID")

rfm_table = rfm_table.rename(columns = {"TotalPrice_x":"Monetary",
                                        "InvoiceDate": "Recency",
                                        "TotalPrice_y": "Frequency"})

# dividing it into 5 sections in terms of customer frequency, monetary, timeliness and finding the RFM score
rfm_table = rfm_table.sort_values("Recency",ascending=True)
rfm_table["Rec_Tile"] = pd.qcut(rfm_table.Recency, 5 , labels = [1,2,3,4,5])
rfm_table["Mone_Tile"] = pd.qcut(rfm_table.Monetary, 5, labels = [1,2,3,4,5])

def FScore(x,p,d):
    if x <= d[p][0.2]:
        return 1
    elif x <= d[p][0.4]:
        return 2
    elif x <= d[p][0.6]:
        return 3
    elif x <= d[p][0.8]:
        return 4
    else:
        return 5

quantiles = rfm_table.quantile(q = [0.2,0.4,0.6,0.8]).to_dict()
rfm_table["Freq_Tile"] = rfm_table["Frequency"].apply(FScore,args=("Frequency",quantiles))
rfm_table["RFM_Score"] = rfm_table["Rec_Tile"].astype("str") + rfm_table["Mone_Tile"].astype("str") + rfm_table["Freq_Tile"].astype("str")
# the best costumers
best = rfm_table[rfm_table.RFM_Score == "555"]
# the worst costumers
worse = rfm_table[rfm_table.RFM_Score == "111"]

# Requency, Monetary, Frequency visualization of customers divided into groups
for col in rfm_table.iloc[:,3:6]:
    plt.figure(figsize=(10,5))
    sns.countplot(rfm_table[col]).set_title(col +" of buy of products by Customers")
    plt.show()

RFM_Score_Group = rfm_table.groupby("RFM_Score").agg({"Recency":["mean","min","max","count"],
                                    "Monetary": ["mean","min","max","count"],
                                    "Frequency":["mean","min","max","count"]}).round(1)

#print(RFM_Score_Group)

# Recency, Monetary, Frequency of their density visualization
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(25,5))
fig.suptitle("RFM Table")
sns.distplot(rfm_table.Recency, ax = axes[0]).set_title("Recency")
sns.distplot(rfm_table.Monetary, ax = axes[1]).set_title("Monetary")
sns.distplot(rfm_table.Frequency, ax = axes[2]).set_title("Frequency")
    

# I have normalized the values (in order not to distort the distance calculations)
clusDataFrame = rfm_table[["Recency","Monetary","Frequency"]]
min_max_scaler = MinMaxScaler()
clusterData = min_max_scaler.fit_transform(clusDataFrame)
clusterData = pd.DataFrame(clusterData)
print(clusterData.describe())

# Determining the best cluster value for the KMeans Algorithm
# Inertia: Bir küme içindeki noktaların ne kadar uzakta oldugunu söyler. Bu nedenle küçük bir inertia hedeflenir.Inertia degeri sıfırdan başlar ve artar.
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init = "k-means++", n_init=10, max_iter=100)
    kmeans.fit(clusterData)
    wcss.append(kmeans.inertia_)
plt.figure(figsize=(10,5))
plt.plot(range(1,11),wcss)
plt.ylabel("WCSS")
plt.xlabel("N-Clusters")
plt.title("Number of Clusters")
plt.show()


# silhouette_score => Bir kümedeki veri noktalarının, başka bir kümedeki veri noktalarından ne kadar uzakta olduğunu gösterir. 1' yakın olması iyi
silhouette_score_list = []
for i in range(2,10):
    kmeans = KMeans(n_clusters=4, init="k-means++", n_init=10, max_iter=100)
    kmeans.fit(clusterData)
    pred = kmeans.predict(clusterData)
    silhouette_score_list.append(silhouette_score(clusterData,kmeans.labels_))
    print(silhouette_score_list)


clusDataFrame["cluster"] = pred
plt.figure(figsize=(10,5))
sns.countplot(clusDataFrame.cluster)
plt.title("Number of Customers by Cluster")
plt.show()

print(clusDataFrame.groupby("cluster").mean())

'''
            Recency     Monetary  Frequency
cluster                                    
0        277.597615   559.960790   1.377049
1        143.763860   886.353349   2.383984
2        595.824324   349.446486   1.513514
3         33.275296  2919.628489   5.793814

Buradan çıkan sonuçlara göre 3 numaralı cluster'da costumer 2919 tl'lik alışveriş yapmış, en son 33 gün önce ve 5 ürün satın almış gibi sonuçlar çıkarabiliriz.,
2 numaralı cluster ise 349 tl'lik alışveriş yapmış, en son 595 gun once ve 1 urun satın almış
Costumers best'den worse'ye doğru sıralayacak olursak 3 > 1 > 0 > 2
'''

# Birliktelik Analizi (Association Rules between two product)
data_apriori = df.groupby(["InvoiceNo","Description"])["Quantity"].sum().unstack().reset_index().fillna(0).set_index("InvoiceNo")
def num(x):
    if x <= 0:
        return 0
    elif x >= 1:
        return 1

basket = data_apriori.applymap(num)

# 1.WAY
rule_fp = fpgrowth(basket, min_support=0.02, use_colnames = True)
print(rule_fp)
# 2.WAY (more detailed)
# products that are found together in 40 percent and more
items = apriori(basket, min_support=0.02, use_colnames=True)
# ürün birliktelik kararları sonuçları
rule = association_rules(items, metric = "confidence", min_threshold=0.4) 
print(rule.sort_values("confidence",ascending=False))

'''
antecedents	                                                                consequents
frozenset({'REGENCY CAKESTAND 3 TIER', 'GREEN REGENCY TEACUP AND SAUCER'})	frozenset({'ROSES REGENCY TEACUP AND SAUCER '})

antecedent support	     consequent support	         support           	      confidence	      lift	             leverage	             conviction
0.03832665330661322    	0.05075150300601202	         0.031663326653306616	0.8261437908496734	16.278213292556252	0.029718191392805654	5.459963159401515

- antecedents tüm ürünlerde bulunma yüzdesi 0.03832665330661322
- consequent tüm ürünlerde bulunma yüzdesi 0.05075150300601202	
- Bu iki ürün için degerlendirme yapacak olursak;
- Tüm ürünlerde beraber bulunma yüzdeleri 0.031663326653306616 (support) //  Support = Freq(A,B)/N
- Bu iki ürünün beraber bulunma olasılığı 0.8261437908496734 (confidence) // Confidence = Freq(A,B)/Freq(A)
- antecedents ürünü olan alışverişlerde consequent de bulunma yüzdesi 16.278213292556252  (lift)


'''


