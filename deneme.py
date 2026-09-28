import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch_geometric.data import Data
import matplotlib.pyplot as plt
from d3graph import d3graph, vec2adjmat
import networkx as nx
import torch
from torch_geometric.loader import LinkNeighborLoader
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv
from sklearn.metrics import average_precision_score, roc_auc_score,classification_report, confusion_matrix
from tqdm import tqdm


matplotlib.use("Qt5Agg")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)


df_acc=pd.read_csv("AML/HI-Small_accounts.csv")
df_trans=pd.read_csv("AML/HI-Small_Trans.csv")


#1. Hesaplar Tablosu (df_acc / Node Verisi)-Bu tablo grafın düğümlerini (Nodes) ve düğüm özniteliklerini (node features) tanımlar.
# Bank Name:Hesabın bağlı olduğu bankanın adı (ör. Portugal Bank #4507).
# Bank ID:Bankanın sayısal benzersiz kimlik kodu.Account Number
# Anahtar Değişken (Düğüm ID'si): Hesabın benzersiz kimliğidir (ör. 80B779D80). Transfer tablosundaki hesaplarla bu ID üzerinden eşleşir.
# Entity ID:Hesabın sahibi olan kişi veya kurumun benzersiz kimlik numarası. Tek bir kurumun birden fazla hesabı olabilir.
# Entity Name:Hesap sahibinin türü/unvanı (ör. Corporation #33520, Sole Proprietorship, Partnership vb.). Tüzel/şahıs ayrımını gösterir.


#2. Transferler Tablosu-Bu tablo grafın yönlü kenarlarını (Directed Edges) ve kenar özniteliklerini (edge features) tanımlar.
#Timestamp:İşlemin gerçekleştiği tarih ve saat. Zaman serisi ve sıralı akış analizi için kullanılır.
# From Bank:Parayı gönderen tarafın banka ID kodu.
# Account:Kaynak Düğüm (Source Node): Parayı gönderen hesabın ID'si.
# To Bank:Parayı alan tarafın banka ID kodu.
# Account.1:Hedef Düğüm (Target Node): Paranın ulaştığı alıcı hesabın ID'si.
# Amount Received:Alıcı hesaba geçen para miktarı.
# Receiving Currency:Alıcı hesabın para birimi (ör. US Dollar).
# Amount Paid:Gönderen hesaptan çıkan para miktarı (Döviz transferlerinde kur farkından dolayı Amount Received ile farklı olabilir).
# Payment Currency:Gönderen tarafın ödeme yaptığı para birimi.
# Payment Format:Transferin yapılış yöntemi/kanalı (ör. Reinvestment, Cheque, Credit Card, Wire/EFT, ACH vb.).
# Is Laundering:Hedef Değişken :• 0: Normal / Yasal işlem• 1: Kara Para Aklama / Dolandırıcılık (Fraud) işlemi.



def get_information(dataframe):
    print(f"dataframe'in boyut bilgileri = \n {dataframe.shape}")
    print("###################################################################################")
    print(f"dataframe'in benzersiz değer sayısı = \n {dataframe.nunique()}")
    print("###################################################################################")
    print(f"dataframe'in boş değer sayısı = \n {dataframe.isnull().sum()}")
    print("###################################################################################")
    print(f"dataframe'in sayısal değerlerinin bilgileri= \n {dataframe.describe().T}")
    print("###################################################################################")
    print(f"dataframe'in ilk 5 gözlemi = \n {dataframe.head()}")
    print("###################################################################################")
    print(f"dataframe'in son 5 gözlemi \n {dataframe.tail()}")
    print("###################################################################################")


#############################################
#  Farklı bankadaki kesişen idlerin yerine benzersiz idler atama
#############################################
df_acc["Bank ID"] = df_acc["Bank ID"].astype(str)
df_trans["From Bank"]=df_trans["From Bank"].astype(str)
df_trans["To Bank"]=df_trans["To Bank"].astype(str)

duplicate_accounts = df_acc[df_acc["Account Number"].duplicated(keep=False)]

#farklı bankalardaki account number çakışmalarını önlemek için
df_acc["Account Number"] = (
    df_acc["Bank ID"].astype(str) + "_" + df_acc["Account Number"].astype(str)
)

# 2. df_trans tablosundaki kaynak ve hedef hesapları güncelle
df_trans["Account"] = (
    df_trans["From Bank"].astype(str) + "_" + df_trans["Account"].astype(str)
)
df_trans["Account.1"] = (
    df_trans["To Bank"].astype(str) + "_" + df_trans["Account.1"].astype(str)
)


#############################################
#    DEĞİŞKENLERİ AYIRMA
#############################################
def grab_col_names(dataframe,cat_th=7,car_th=20):
    num_cols=[col for col in dataframe.columns if dataframe[col].dtypes != "str"]
    cat_cols=[col for col in dataframe.columns if dataframe[col].dtypes == "str"]
    num_but_cat=[col for col in dataframe.columns if dataframe[col].dtypes != "str" and dataframe[col].nunique() < cat_th]
    cat_but_car=[col for col in dataframe.columns if dataframe[col].dtypes == "str" and dataframe[col].nunique() > car_th]

    cat_cols=cat_cols+num_but_cat
    cat_cols=[col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols_shape: {len(cat_cols)} , cat_cols = {cat_cols}')
    print(f'num_cols_shape: {len(num_cols)} , num_cols = {num_cols}' )
    print(f'cat_but_car_shape: {len(cat_but_car)} , cat_but_car = {cat_but_car}')
    print(f'num_but_cat_shape: {len(num_but_cat)} , num_but_cat = {num_but_cat}')

    return cat_cols,num_cols,cat_but_car,num_but_cat

cat_cols_acc,num_cols_acc,cat_but_car_acc,num_but_cat_acc = grab_col_names(df_acc)

cat_cols_trans,num_cols_trans,cat_but_car_trans,num_but_cat_trans = grab_col_names(df_trans)

num_cols_trans=[col for col in num_cols_trans if col not in ["From Bank","To Bank","Timestamp"]]



def cat_sumary(dataframe):
    """

    :param dataframe: burada kullanılacak olan dataframe yazılır
    :return: dönen değer ise kategorik değişken analizi için görselleştirmedir
    """


#############################################
#KATEGORİK DEĞİŞKEN ANALİZİ
#############################################
def cat_summary(dataframe,col_name,plot=False):

    print(pd.DataFrame({col_name:dataframe[col_name].value_counts(),
                 "ratio":dataframe[col_name].value_counts() * 100 / dataframe.shape[0]} ))

    if plot:
        plt.figure(figsize=(10, 5))  #çizilecek graafiğin alanını hazırla
        sns.countplot(data=dataframe, x=col_name)    #grafiği çiz
        plt.title(f"Distribution of {col_name}")
        plt.xticks(rotation=45)   #X eksenindeki kategori isimlerini 45 derece döndürüyor.
        plt.tight_layout()        #Grafiğin elemanlarının birbirinin veya pencerenin dışına taşmasını önlemeye çalışıyor.

        # Pencere açılır; kapatana kadar kod burada bekler (bloklar)
        plt.show()



for col in cat_cols_trans:
    cat_summary(df_trans, col , plot=True)


#############################################
#SAYISAL DEĞİŞKEN ANALİZİ
#############################################
def num_summary(dataframe,numerical_col,plot=False):

    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]

    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        plt.figure(figsize=(10,5))
        plt.hist(np.log1p(dataframe[numerical_col]),bins=50)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.tight_layout()
        plt.show()

for col in num_cols_trans:
    num_summary(df_trans,col,plot=True)



df_trans[(df_trans["Amount Paid"]== 0) & (df_trans["Amount Received"]== 0)].index
print("Paid = 0:", (df_trans["Amount Paid"] == 0).sum())
print("Received = 0:", (df_trans["Amount Received"] == 0).sum())



######################################
# 5. Korelasyon Analizi (Analysis of Correlation)
######################################

def corr(dataframe):
    # Korelasyon Matrisi
    numeric_df=dataframe.select_dtypes(include="number")
    f, ax = plt.subplots(figsize=[18, 13])
    sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
    ax.set_title("Correlation Matrix", fontsize=20)
    plt.show(block=True)

corr(df_trans)

######################################
# Feature Extraction
######################################

def common_feature_Extraction(dataframe):
    dataframe["Timestamp"]=pd.to_datetime(dataframe["Timestamp"])   #tarih değişkenini datatime değişkenine dönüştürüp özellik üreticez

    dataframe["day"] = dataframe["Timestamp"].dt.day
    dataframe["hour"] = dataframe["Timestamp"].dt.hour
    dataframe["dayofweek"] = dataframe["Timestamp"].dt.dayofweek
    dataframe["minute"] = dataframe["Timestamp"].dt.minute

    dataframe["hour_sin"] = np.sin(2 * np.pi * dataframe["hour"] / 24)
    dataframe["hour_cos"] = np.cos(2 * np.pi * dataframe["hour"] / 24)

    # Minute
    dataframe["minute_sin"] = np.sin(2 * np.pi * dataframe["minute"] / 60)
    dataframe["minute_cos"] = np.cos(2 * np.pi * dataframe["minute"] / 60)


    # Day of week (0-6 ise)
    dataframe["dayofweek_sin"] = np.sin(2 * np.pi * dataframe["dayofweek"] / 7)
    dataframe["dayofweek_cos"] = np.cos(2 * np.pi * dataframe["dayofweek"] / 7)

    #day
    dataframe["day_sin"] = np.sin(2 * np.pi * dataframe["day"] / 30)
    dataframe["day_cos"] = np.cos(2 * np.pi * dataframe["day"] / 30)

common_feature_Extraction(df_trans)

def discrete_feature_extraction(dataframe):
    dataframe=dataframe.copy()
    dataframe = dataframe.sort_values("Timestamp")

# burada shift bir kaydırma yapar nasıl yani:
    #10
    #20
    #30
#ise shift yapınca artık bu :
    #NaN
    #10
    #20
#olur ve expanding yapınca kendisine kadar olan kısmı alır
    # NaN
    # 10     #[10]
    # 20     #[10,20]
#sonra bunlara mean uygulanır böylece model ileri tarihli değerleri görmemiş olur

    # Transaction Counts
    dataframe["Number of Transactions Sent"] = (
        dataframe.groupby("Account")["Account"]
        .transform(lambda x: x.shift().expanding().count())
    )

    dataframe["Number of Transactions Received"] = (
        dataframe.groupby("Account.1")["Account.1"]
        .transform(lambda x: x.shift().expanding().count())
    )

    # Historical Total Amounts
    dataframe["Total Amount Paid"] = (
        dataframe.groupby("Account")["Amount Paid"]
        .transform(lambda x: x.shift().expanding().sum())
    )

    dataframe["Total Amount Received"] = (
        dataframe.groupby("Account.1")["Amount Received"]
        .transform(lambda x: x.shift().expanding().sum())
    )

    # Historical Means
    dataframe["Mean Amount Paid"] = (
        dataframe.groupby("Account")["Amount Paid"]
        .transform(lambda x: x.shift().expanding().mean())
    )

    dataframe["Mean Amount Received"] = (
        dataframe.groupby("Account.1")["Amount Received"]
        .transform(lambda x: x.shift().expanding().mean())
    )

    # Mean Ratios
    dataframe["Paid Mean Ratio"] = (
            dataframe["Amount Paid"] /
            dataframe["Mean Amount Paid"]
    )

    dataframe["Received Mean Ratio"] = (
            dataframe["Amount Received"] /
            dataframe["Mean Amount Received"]
    )

    # Historical Medians
    dataframe["Median Amount Paid"] = (
        dataframe.groupby("Account")["Amount Paid"]
        .transform(lambda x: x.shift().expanding().median())
    )

    dataframe["Median Amount Received"] = (
        dataframe.groupby("Account.1")["Amount Received"]
        .transform(lambda x: x.shift().expanding().median())
    )

    # Median Ratios
    dataframe["Paid Median Ratio"] = (
            dataframe["Amount Paid"] /
            dataframe["Median Amount Paid"]
    )

    dataframe["Received Median Ratio"] = (
            dataframe["Amount Received"] /
            dataframe["Median Amount Received"]
    )

    # Historical Balance
    dataframe["Total Received - Total Paid"] = (
            dataframe["Total Amount Received"]
            - dataframe["Total Amount Paid"]
    )

    # Historical Bank Transaction Counts
    dataframe["Transaction Count From Bank"] = (
        dataframe.groupby("From Bank")["From Bank"]
        .transform(lambda x: x.shift().expanding().count())
    )

    dataframe["Transaction Count To Bank"] = (
        dataframe.groupby("To Bank")["To Bank"]
        .transform(lambda x: x.shift().expanding().count())
    )

    # Bank Ratio
    dataframe["Bank Transaction Count Ratio"] = (
            dataframe["Transaction Count From Bank"]
            / dataframe["Transaction Count To Bank"]
    )

    # Historical Unique Receivers


    # Unique Receiver Ratio
    # Account ve Account.1 kolonlarını sayısal koda çevir (nunique için gerekli)
    dataframe["Account_code"] = pd.factorize(dataframe["Account"])[0]
    dataframe["Account1_code"] = pd.factorize(dataframe["Account.1"])[0]

    # Historical Unique Receivers
    dataframe["Unique Receivers"] = (
        dataframe.groupby("Account")["Account1_code"]
        .transform(
            lambda x: x.shift().expanding().apply(
                lambda y: pd.Series(y).nunique(),
                raw=True  # artık numpy array geliyor, raw=True daha hızlı
            )
        )
    )

    # Historical Unique Senders
    dataframe["Unique Senders"] = (
        dataframe.groupby("Account.1")["Account_code"]
        .transform(
            lambda x: x.shift().expanding().apply(
                lambda y: pd.Series(y).nunique(),
                raw=True
            )
        )
    )

    dataframe = dataframe.drop(columns=["Account_code", "Account1_code"])


    dataframe["Is Same Account"] = (dataframe["Account"] == dataframe["Account.1"]).astype(int)
    dataframe["Is Same Bank"] = (dataframe["From Bank"] == dataframe["To Bank"]).astype(int)

    return dataframe

df_trans=discrete_feature_extraction(df_trans)
df_trans.tail()

df_acc["Entity Name"].str.contains("#").sum()
df_acc["Entity Type"] = df_acc["Entity Name"].str.extract(r"^(.*?)\s*#")

"""
df_trans.head()
df_trans.drop("Timestamp",axis=1,inplace=True)
df_trans.drop("day",axis=1,inplace=True)
df_trans.drop("hour",axis=1,inplace=True)
df_trans.drop("dayofweek",axis=1,inplace=True)
df_trans.drop("minute",axis=1,inplace=True)
"""

df_trans.head()


######################################
#NaN değerleri 0 ile doldurma
######################################

df_trans=df_trans.fillna(0)


######################################
# Rare Analizi
######################################

def rare_analyser(dataframe,cat_cols):
    for col in cat_cols:
        print(pd.DataFrame({"count": dataframe[col].value_counts(),
                            "ratio": dataframe[col].value_counts() * 100 / len(dataframe)}))


rare_analyser(df_trans,cat_cols_trans)

#oranlar yeterli büyüklükte olduğu için rare encoder yapma ihtiyacında bulunmadım


df_trans_display = df_trans.copy()


######################################
# One Hot Encoder
######################################
def one_hot_encoder(dataframe,ohe_cols_,drop_first=True):
    dataframe=pd.get_dummies(dataframe, columns=ohe_cols_, drop_first=drop_first,dtype=int)   #get_dummies yeni df döndürür
    return dataframe



ohe_cols=[col for col in cat_cols_trans if col not in ["Is Laundering"]]


df_trans=one_hot_encoder(df_trans,ohe_cols,drop_first=True)

df_trans.head()
df_trans.tail()

cat_cols_acc,num_cols_acc,cat_but_car_acc,num_but_cat_acc = grab_col_names(df_acc)

df_acc=one_hot_encoder(df_acc,cat_cols_acc,drop_first=True)

df_acc.head()

######################################
#df_acc analizi
######################################
for col in cat_cols_acc:
    cat_summary(df_acc,col,True)

for col in num_cols_acc:
    num_summary(df_acc,col,True)

get_information(df_acc)
get_information(df_trans)


######################################
#true/false -->0/1
######################################

for col in df_trans.columns:
    if df_trans[col].dtype == "bool":
        df_trans[col] = df_trans[col].astype(int)

df_trans.tail()

for col in df_acc.columns:
    if df_acc[col].dtype == "bool":
        df_acc[col] = df_acc[col].astype(int)


######################################
#log dönüşümü
######################################

cat_cols_trans,num_cols_trans,cat_but_car_trans,num_but_cat_trans = grab_col_names(df_trans)
num_cols_trans=[col for col in num_cols_trans if col not in "Timestamp" ]

num_summary(df_trans,num_cols_trans,plot=False)

#### infleri yok etme yani 0 a bölme hatasını yok etme
df_trans["Bank Transaction Count Ratio"]=np.where(df_trans["Transaction Count To Bank"] == 0,
                                             0,
                                             df_trans["Transaction Count From Bank"] / df_trans["Transaction Count To Bank"])
#np.where numpyın if elsesi gibi Transaction Count To Bank 0 sa Bank Transaction Count Ratio 0 yaz değilse bölmeyi yap yaz


skew_values = df_trans[num_cols_trans].skew().sort_values(ascending=False)  #nümerik kolonların çarpıklık değerlerini kontrol etme
print(skew_values)


continuous_cols = [
    "Amount Paid", "Amount Received",
    "Total Amount Paid", "Total Amount Received",
    "Mean Amount Paid", "Mean Amount Received",
    "Median Amount Paid", "Median Amount Received",
    "Number of Transactions Sent", "Number of Transactions Received",
    "Transaction Count From Bank", "Transaction Count To Bank",
    "Unique Receivers", "Unique Senders",
    "Paid Mean Ratio", "Received Mean Ratio",
    "Paid Median Ratio", "Received Median Ratio",
    "Bank Transaction Count Ratio",
    "Total Received - Total Paid"
]

df_trans.info()


# 1. Hepsine tek seferde Signed Log uygulama
df_trans[continuous_cols] = np.sign(df_trans[continuous_cols]) * np.log1p(np.abs(df_trans[continuous_cols]))
# bu işleme simetrik log dönüşümü denir neden direk log1p değil çünkü bazı sütunların değerleri - değer - değere log dönüşümü uygulanmaz bu yüzden np.sign sayı - ise -1 + ise +1 döndürür
#np.abs(df_trans[continuous_cols]) sayının mutlak değerini alıp log uygular daha sonra sign ile çarpılınca zaten sayı - ise - , + ise + döner



######################################
#ölçekleme
######################################
#ölçekleme yapmadan önce test train ayrımı yapmamız zorunlu çünkü ölçeklerken ortalama değerlerini kullanıyr

df_trans = df_trans.sort_values(
    "Timestamp",
    kind="stable"
).reset_index(drop=True)


df_trans_display = df_trans_display.sort_values(
    "Timestamp",
    kind="stable"
).reset_index(drop=True)
train_df,test_df=train_test_split(df_trans,test_size=0.2,train_size=0.8,shuffle=False)

train_df=train_df.copy().reset_index(drop=True)
test_df=test_df.copy().reset_index(drop=True)

scaler=StandardScaler()

train_df[continuous_cols]=scaler.fit_transform(train_df[continuous_cols])
test_df[continuous_cols]=scaler.transform(test_df[continuous_cols])

df_scaled = pd.concat([train_df, test_df], ignore_index=True) #ignore_index indeexleri 0lar

# 5. Maskeleri ekle
df_scaled["train_mask"] = [True] * len(train_df) + [False] * len(test_df)
df_scaled["test_mask"] = [False] * len(train_df) + [True] * len(test_df)
#bu satırların amacı model tahmin hepsi ile tahmin yapsın ama parametrelerini sadece trainden güncellesin

df_scaled.head()

all_unique_accounts = pd.concat([
    df_acc["Account Number"],
    df_scaled["Account"],
    df_scaled["Account.1"]
]).unique()



# 2. String ID -> Tamsayı İndeks sözlüğü (Mapping Dictionary)
account_to_idx = {acc: idx for idx, acc in enumerate(all_unique_accounts)}
num_nodes = len(all_unique_accounts)
print(f"Toplam Tekil Düğüm (Hesap) Sayısı: {num_nodes}")



# Kaynak ve hedef hesapları ID sözlüğünden geçir
src_nodes = df_scaled["Account"].map(account_to_idx).values
dst_nodes = df_scaled["Account.1"].map(account_to_idx).values
#mapin özelliği içine dict alabilmesidir mesela account tablosunu dicte göre güncelliyor yani account diyelim ki a
#ve account_to_idx dictinde eşleşme a:1 şeklinde ise artık account tablosu 1 oluyor



# edge_index tensörü: Shape [2, Num_Edges]
edge_index = torch.tensor(np.array([src_nodes, dst_nodes]), dtype=torch.long)
print(f"edge_index boyutu: {edge_index.shape}")




# ----------------- DÜĞÜM ÖZELLİKLERİ (x) -----------------
exclude_acc_cols = ["Account Number", "Bank Name", "Bank ID", "Entity ID", "Entity Name"]
node_feature_cols = [col for col in df_acc.columns if col not in exclude_acc_cols]

# Tüm hesapları içeren boş bir iskelet oluşturup df_acc ile merge ediyoruz
# (df_trans'ta olup df_acc'de olmayan hesaplar varsa 0 ile doldurulur)
node_features_df = pd.DataFrame({"Account Number": all_unique_accounts})
node_features_df = node_features_df.merge(
    df_acc[["Account Number"] + node_feature_cols],
    on="Account Number",
    how="left"
).fillna(0)

x = torch.tensor(node_features_df[node_feature_cols].values, dtype=torch.float)

# ----------------- KENAR ÖZELLİKLERİ (edge_attr) -----------------
exclude_edge_cols = [
    "Timestamp", "From Bank", "To Bank", "Account", "Account.1",
    "Is Laundering", "train_mask", "test_mask",
    "day", "hour", "dayofweek", "minute"
]
edge_feature_cols = [col for col in df_scaled.columns if col not in exclude_edge_cols]

edge_attr = torch.tensor(df_scaled[edge_feature_cols].values, dtype=torch.float)

# ----------------- HEDEF VE MASKELER -----------------
y = torch.tensor(df_scaled["Is Laundering"].values, dtype=torch.float)
train_mask = torch.tensor(df_scaled["train_mask"].values, dtype=torch.bool)
test_mask = torch.tensor(df_scaled["test_mask"].values, dtype=torch.bool)



graph_data = Data(
    x=x,
    edge_index=edge_index,
    edge_attr=edge_attr,
    y=y,
    train_mask=train_mask,
    test_mask=test_mask
)

print("\n--- Graf Nesnesi Başarıyla Oluşturuldu ---")
print(graph_data)


#############################################
#GRAPH GÖRSELLEŞME
#############################################
# 1. Her hesabın ulaştığı en yüksek işlem sayısını hesapla
max_islemler = df_trans_display.groupby("Account")["Number of Transactions Sent"].max()
farklı_hesap=df_trans_display.groupby("Account")["Account.1"].nunique()

# 2. Maksimumu tam 20 olan hesapların ID'lerini liste olarak al
hesaplar_20 = max_islemler[(max_islemler >= 40) & (farklı_hesap>15)].index.tolist()
hesaplar_20[0:20]

# İlk 5 tanesini gör
print(hesaplar_20[:5])
#10057_803D912B0


"""
def tek_hop(node):
    src_nodes_ = df_trans_display.loc[df_trans_display["Account"]=="10_800055DF0","Account"].map(account_to_idx).values
    dst_nodes_ = df_trans_display.loc[df_trans_display["Account"]=="10_800055DF0","Account.1"].map(account_to_idx).values
    weight_=df_trans_display.loc[df_trans_display["Account"]=="10_800055DF0","Amount Received"].values

    return src_nodes_,dst_nodes_,weight_

src_nodes,dst_nodes,weight=tek_hop("10_800055DF0")

def graph_ciz(src_nodes_,dst_nodes_,weight_):
    d3=d3graph()
    adjmat=vec2adjmat(src_nodes_, dst_nodes_, weight=weight_)
    d3.graph(adjmat)
    d3.set_node_properties(tooltip=["Hesap ID: " + str(node) for node in d3.node_properties.keys()])
    d3.set_edge_properties(directed=True)
    d3.show()

"""



G = nx.from_pandas_edgelist(df_trans_display,source="Account",target="Account.1",edge_attr="Amount Received",create_using=nx.DiGraph())
#dict formatında komşuluk ilişkileri çıkar örneğin
#G._adj = {
    #'A': {'B': {}, 'C': {}},
    #'B': {'A': {}, 'D': {}, 'E': {}},
    #'C': {'A': {}, 'F': {}},


"""
# 2. Dinamik K-Hop Çizdirme Fonksiyonu
def k_hop_ciz(node_id, k_hop=1):
    if node_id not in G:
        print(f"{node_id} graf içinde bulunamadı!")
        return

    # NetworkX arka planda BFS ile döngülere takılmadan k derinliğe kadar iner
    sub_G = nx.ego_graph(G, n=node_id, radius=k_hop)
    #artık burada çizilecek alt graph çıkar yani {A:0,B:1,C:1,D:2,E:2,F:2} gibi bir çıktı oldu daha sonra zaten bu
    #yukarıdaki G ye bakılarak rahatça çizilebilir

    # Subgraph'ı d3graph'ın doğrudan okuyabileceği adjacency matrisine çevir
    adjmat = nx.to_pandas_adjacency(sub_G, weight="Amount Received")
    #daha sonra bu sub_G de kalanlari için komşuluk matrisi üretilir satıra A B C D E F sütuna da aynısı yzaılır
    #kesişimlerine ise  weight="Amount Received" bakılarak değerleri yazılır ve d3 graphla görselleştirlir

    # Görselleştirme
    d3 = d3graph()
    d3.graph(adjmat)
    d3.set_node_properties(tooltip=["Hesap ID: " + str(node) for node in d3.node_properties.keys()])
    d3.set_edge_properties(directed=True, min_weight=0)
    d3.show()

secilen_hesap = hesaplar_20[0]

mesafeler = nx.single_source_shortest_path_length(G, source=secilen_hesap)

# mesafeler sözlüğündeki en büyük değer (en uzak hop):
max_hop_node = max(mesafeler.values())
en_uzak_hesap = [k for k, v in mesafeler.items() if v == max_hop_node]

print(f"{secilen_hesap} kaynağından para en fazla {max_hop_node} hop uzağa gidebiliyor.")

print(f"En uçtaki hesap(lar): {en_uzak_hesap}")

# 1 hop (Sadece doğrudan alıcılar)
k_hop_ciz(secilen_hesap, k_hop=1)

# 2 hop (Alıcılar + onların alıcıları)
k_hop_ciz(secilen_hesap, k_hop=2)

# 3 hop
k_hop_ciz(secilen_hesap, k_hop=max_hop_node)

"""

def parse_pattern_from_text(txt_path, pattern_index=0):
    with open(txt_path, "r", encoding="utf-8") as f:
        content = f.read()

    blocks = content.strip().split("END LAUNDERING ATTEMPT")
    selected_block = blocks[pattern_index]

    parsed_edges = []
    for line in selected_block.splitlines():
        line = line.strip()
        if not line or line.startswith("BEGIN") or "," not in line:
            continue

        parts = line.split(",")
        if len(parts) >= 6:
            from_bank = str(int(parts[1]))
            to_bank = str(int(parts[3]))

            src = f"{from_bank}_{parts[2]}"
            dst = f"{to_bank}_{parts[4]}"
            amount = float(parts[5])
            parsed_edges.append((src, dst, amount))

    return parsed_edges


def visualize_pattern_subgraph(G, pattern_edges, include_context_hops=0, max_neighbors_per_node=3):
    pattern_nodes = {src for src, dst, _ in pattern_edges} | {dst for src, dst, _ in pattern_edges}

    if include_context_hops == 0:
        sub_nodes = pattern_nodes
    else:
        sub_nodes = set(pattern_nodes)
        for node in pattern_nodes:
            if node in G:
                # Düğüm patlamasını önlemek için sınırlı sayıda komşu al
                in_n = list(G.predecessors(node))[:max_neighbors_per_node]
                #G.predecessors verilen node'a giren düğümleri verir
                out_n = list(G.successors(node))[:max_neighbors_per_node]
                # G.successors verilen node'dan çıkan düğümleri verir
                sub_nodes.update(in_n + out_n)

    print(f"Görselleştirilecek Düğüm Sayısı: {len(sub_nodes)}")

    sub_G = G.subgraph(sub_nodes).copy()
    #G.subgraph ana bağlantıların olduğu dictten sadece sub_nodes olanları çeker

    adjmat = nx.to_pandas_adjacency(sub_G, weight="Amount Received")

    d3 = d3graph(charge=-1100, collision=2.0)
    d3.graph(adjmat)

    node_colors = ['#E74C3C' if n in pattern_nodes else '#3498DB' for n in d3.node_properties.keys()]

    d3.set_node_properties(
        color=node_colors,
        label="",
        tooltip=["Hesap: " + str(n) + (" [FRAUD PATTERN]" if n in pattern_nodes else " [NORMAL]") for n in
                 d3.node_properties.keys()]
    )
    d3.set_edge_properties(
        directed=True,
        min_weight=0,
        edge_distance=250 # Bağlantı mesafesini uzat (Varsayılan: 50-100)
    )

    "BURAYA RENK DEĞİŞİMİ YAZILACAK"
    for src, dst, _ in pattern_edges:
        if (src, dst) in d3.edge_properties:
            d3.edge_properties[(src, dst)]['edge_color'] = '#9b0635'

    d3.show()


# Örnek Kullanım:
# 1. TXT dosyasındaki ilk deseni (örneğin FAN-OUT veya CYCLE) çek
pattern_edges = parse_pattern_from_text("AML/HI-Small_Patterns.txt", pattern_index=45)

# Sadece deseni izole görmek için:
visualize_pattern_subgraph(G, pattern_edges, include_context_hops=2)

# Desenin etrafındaki hesaplarla ilişkisini görmek için (1 hop bağlam):
# visualize_pattern_subgraph(G, pattern_edges, include_context_hops=1)

df_trans_display.head()


###################################
# MODEL KURULUM AŞAMALARI: DATA LOADER (TEMPORAL ISOLATION)
###################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Kullanılan Donanım: {device}")

# Kenar indekslerini çekiyoruz
train_edge_indices = torch.where(graph_data.train_mask)[0]
test_edge_indices = torch.where(graph_data.test_mask)[0]

# Train esnasında mesaj iletiminin akacağı izole graf kenarları
train_edge_index = graph_data.edge_index[:, train_edge_indices]
train_edge_attr = graph_data.edge_attr[train_edge_indices]

# Train Dataset & Loader: Sadece tahmin edilecek kenarların batch'leri üretilir
train_graph_data = Data(
    x=graph_data.x,
    edge_index=train_edge_index,
    edge_attr=train_edge_attr
)
train_graph_data.edge_label_index = train_edge_index
train_graph_data.edge_label = graph_data.y[train_edge_indices]

train_loader = LinkNeighborLoader(
    train_graph_data,
    num_neighbors=[20,20],
    edge_label_index=train_graph_data.edge_label_index,
    edge_label=train_graph_data.edge_label,
    batch_size=2048,
    shuffle=True,
    disjoint=True,
    subgraph_type="directional"
)

# Test Dataset & Loader
test_graph_data = Data(
    x=graph_data.x,
    edge_index=train_edge_index,
    edge_attr=train_edge_attr
)
test_graph_data.edge_label_index = graph_data.edge_index[:, test_edge_indices]
test_graph_data.edge_label = graph_data.y[test_edge_indices]

test_loader = LinkNeighborLoader(
    test_graph_data,
    num_neighbors=[20,20],
    edge_label_index=test_graph_data.edge_label_index,
    edge_label=test_graph_data.edge_label,
    batch_size=4096,
    shuffle=False,
    disjoint=True,
    subgraph_type="directional"
)

print(f"Toplam Train Batch Sayısı: {len(train_loader)}")
print(f"Toplam Test Batch Sayısı: {len(test_loader)}")


###################################
# 2. MODEL MİMARİSİ
###################################
class AML_EdgeGNN(nn.Module):
    def __init__(self, in_node_feats, in_edge_feats, hidden_dim=64):
        super().__init__()
        self.node_encoder = nn.Linear(in_node_feats, hidden_dim)
        self.edge_encoder = nn.Linear(in_edge_feats, hidden_dim)

        # 1. Katman
        mlp1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.conv1 = GINEConv(mlp1, edge_dim=hidden_dim)

        # 2. Katman
        mlp2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.conv2 = GINEConv(mlp2, edge_dim=hidden_dim)

        # Sınıflandırıcı: Kaynak (64) + Hedef (64) + Kenar (64) = 192
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 5, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1)
        )

    def get_node_embeddings(self, x, edge_index, edge_attr):
        # Mesaj iletimi tüm graf üzerinde 1 kez koşar
        h = self.node_encoder(x)
        e = self.edge_encoder(edge_attr)
        h = F.relu(self.conv1(h, edge_index, edge_attr=e))
        h = self.conv2(h, edge_index, edge_attr=e)
        return h

    def predict_edges(self, h, batch_edges, batch_edge_attr):
        # h: [Num_Nodes, 64]
        # batch_edges: [2, Batch_Size]
        src_nodes = batch_edges[0]
        dst_nodes = batch_edges[1]

        h_src = h[src_nodes]
        h_dst = h[dst_nodes]
        e_batch = self.edge_encoder(batch_edge_attr)

        h_mul = h_src * h_dst
        h_diff = torch.abs(h_src - h_dst)

        edge_repr = torch.cat([h_src, h_dst, h_mul, h_diff, e_batch], dim=-1)
        out = self.classifier(edge_repr)
        return out.squeeze(-1)


# Model İlklendirme
num_node_features = graph_data.x.shape[1]
num_edge_features = graph_data.edge_attr.shape[1]

model = AML_EdgeGNN(
    in_node_feats=num_node_features,
    in_edge_feats=num_edge_features,
    hidden_dim=64
).to(device)


###################################
# 3. KAYIP FONKSİYONU & OPTIMIZER
###################################
train_labels = graph_data.y[graph_data.train_mask]
num_positives = (train_labels == 1).sum().item()
num_negatives = (train_labels == 0).sum().item()

pos_weight_value = 10.0
pos_weight = torch.tensor([pos_weight_value], dtype=torch.float).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)


###################################
# 4. EĞİTİM VE TEST FONKSİYONLARI (HIZLI & CANLI İLERLEME)
###################################
def train_one_epoch(model, dataloader, optimizer, criterion, x, train_edge_index, train_edge_attr, device):
    model.train()
    total_loss = 0.0

    # 1. Adım: Her batch için örneklenmiş graf üzerinde düğüm vektörlerini çıkar
    pbar = tqdm(dataloader, desc="Training Batches", leave=False)
    for batch in pbar:
        batch = batch.to(device)
        batch_edges = batch.edge_label_index # [2, Batch_Size]
        batch_labels = batch.edge_label.to(device).float()
        batch_edge_attr = train_edge_attr[batch.input_id.cpu()].to(device)

        optimizer.zero_grad()

        h = model.get_node_embeddings(batch.x, batch.edge_index, batch.edge_attr)
        out = model.predict_edges(h, batch_edges, batch_edge_attr)
        loss = criterion(out, batch_labels)

        # retain_graph=False (varsayılan) ile temiz backward
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate(model, dataloader, x, full_edge_index, full_edge_attr, device):
    model.eval()
    all_preds = []
    all_labels = []

    # Test anında sadece train grafından örneklenmiş düğüm vektörleri çıkarılır
    for batch in dataloader:
        batch = batch.to(device)
        batch_edges = batch.edge_label_index
        batch_labels = batch.edge_label.to(device).float()
        batch_edge_attr = graph_data.edge_attr[test_edge_indices][batch.input_id.cpu()].to(device)

        h = model.get_node_embeddings(batch.x, batch.edge_index, batch.edge_attr)
        out = model.predict_edges(h, batch_edges, batch_edge_attr)
        probs = torch.sigmoid(out)

        all_preds.extend(probs.cpu().numpy())
        all_labels.extend(batch_labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    pr_auc = average_precision_score(all_labels, all_preds)
    roc_auc = roc_auc_score(all_labels, all_preds)

    return pr_auc, roc_auc


###################################
# 5. EĞİTİM DÖNGÜSÜ (EN İYİ MODELİ OTOMATİK KAYDETME)
###################################
# Ana tensörleri GPU'ya tek seferde taşıyoruz
x_gpu = graph_data.x.to(device)
train_edge_index_gpu = train_edge_index.to(device)
train_edge_attr_gpu = train_edge_attr.to(device)

full_edge_index_gpu = graph_data.edge_index.to(device)
full_edge_attr_gpu = graph_data.edge_attr.to(device)

epochs = 10
best_pr_auc = 0.0  # En iyi skoru takip edeceğimiz değişken

print("\n--- Model Eğitimi Başlıyor ---")

for epoch in range(1, epochs + 1):
    loss = train_one_epoch(
        model, train_loader, optimizer, criterion,
        x_gpu, train_edge_index_gpu, train_edge_attr_gpu, device
    )

    pr_auc, roc_auc = evaluate(
        model, test_loader,
        x_gpu, train_edge_index_gpu, train_edge_attr_gpu, device
    )

    print(f"Epoch {epoch:02d} | Train Loss: {loss:.4f} | Test PR-AUC: {pr_auc:.4f} | Test ROC-AUC: {roc_auc:.4f}")

    # Eğer yeni PR-AUC önceki en iyiden yüksekse diske kaydet
    if pr_auc > best_pr_auc:
        best_pr_auc = pr_auc
        torch.save(model.state_dict(), "best_aml_edge_gnn.pth")
        print(f"   -> [KAYDEDİLDİ] Yeni en iyi model! (PR-AUC: {best_pr_auc:.4f})")

print(f"\nEğitim bitti! Diske kaydedilen en iyi skor: {best_pr_auc:.4f}")


#####################################
#ÇIKTI DEĞERLENDİRME
#####################################

# Diske kaydedilen en yüksek performanslı modeli yüklüyoruz
model.load_state_dict(torch.load("best_aml_edge_gnn.pth"))
model.eval()
all_preds = []
all_labels = []

# Test embedding'lerini çıkar
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        batch_edges = batch.edge_label_index
        batch_edge_attr = graph_data.edge_attr[test_edge_indices][batch.input_id.cpu()].to(device)

        h = model.get_node_embeddings(batch.x, batch.edge_index, batch.edge_attr)
        out = model.predict_edges(h, batch_edges, batch_edge_attr)
        probs = torch.sigmoid(out)

        all_preds.extend(probs.cpu().numpy())
        all_labels.extend(batch.edge_label.cpu().numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# 0.5 eşik değerine (threshold) göre tahminler:
thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
print("\n--- FARKLI THRESHOLD DEĞERLERİNİN KARŞILAŞTIRILMASI ---")
print("Threshold | Precision | Recall | F1")

for threshold in thresholds:
    pred_binary = (all_preds >= threshold).astype(int)

    report = classification_report(
        all_labels,
        pred_binary,
        target_names=["Normal (0)", "Fraud (1)"],
        output_dict=True,
        zero_division=0
    )

    precision = report["Fraud (1)"]["precision"]
    recall = report["Fraud (1)"]["recall"]
    f1 = report["Fraud (1)"]["f1-score"]

    print(f"{threshold:9.2f} | {precision:9.3f} | {recall:6.3f} | {f1:5.3f}")

# Şimdilik mevcut 0.5 threshold ile final confusion matrix:
threshold = 0.5
pred_binary = (all_preds >= threshold).astype(int)

print(f"\n--- TEST KÜMESİ PERFORMANSI (Eşik = {threshold}) ---")
print(classification_report(
    all_labels,
    pred_binary,
    target_names=["Normal (0)", "Fraud (1)"],
    zero_division=0
))

print("Karmaşıklık Matrisi (Confusion Matrix):")
print(confusion_matrix(all_labels, pred_binary))