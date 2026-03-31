# Hangi conda environmentini kullanıyorum.   (alt+shift+e)
conda env list
# Environmentler arasında hangi paketler ulaşılabilir durumda?
conda list
# Projeyi daha sağlıklı çalıştırabilmek için base-environmentları paylaşıyoruz.
conda env export > environment.yaml
# Yeni proje için environment yaratmak için:
conda create -n ACM_476_2025_env
# Environmenti aktif etmek için.
conda activate ACM_476_2025_env
# Proje için basei başlangıç kabul edip üstüne .yaml kuruyoruz.
conda env create -f environment.yaml
# Hangi paketlerin ulaşılabilir olduğuna bakıyoruz.
conda list
# Tüm paketleri güncelliyoruz.
conda upgrade -all

## --------- DATA ANALYSIS -----------##
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

import warnings
# # Suppressing a warning
warnings.filterwarnings("ignore")
warnings.warn("this will not show")

import re
import time
import random
import tempfile
from tqdm.notebook import tqdm

import gc
gc.collect()

#--------------DISPLAYING DATA-----------------#
df_origin_train = pd.read_csv(r"C:\Users\guven\Desktop\Credit score classification\train.csv")
df_train = df_origin_train.copy()
df_train
#-------#
#  ID Customer_ID  ...     Monthly_Balance Credit_Score
#  0       0x1602   CUS_0xd40  ...  312.49408867943663         Good
#  1       0x1603   CUS_0xd40  ...  284.62916249607184         Good
#  2       0x1604   CUS_0xd40  ...   331.2098628537912         Good
#  3       0x1605   CUS_0xd40  ...  223.45130972736786         Good
#  4       0x1606   CUS_0xd40  ...  341.48923103222177         Good
#  ...        ...         ...  ...                 ...          ...
#  99995  0x25fe9  CUS_0x942c  ...          479.866228         Poor
#  99996  0x25fea  CUS_0x942c  ...           496.65161         Poor
#  99997  0x25feb  CUS_0x942c  ...          516.809083         Poor
#  99998  0x25fec  CUS_0x942c  ...          319.164979     Standard
#  99999  0x25fed  CUS_0x942c  ...          393.673696         Poor
#  [100000 rows x 28 columns]
#---------------------------------------------------------------#
df_origin_test = pd.read_csv(r"C:\Users\guven\Desktop\Credit score classification\test.csv")
df_test = df_origin_test.copy()
df_test
#------#
#              ID  ...     Monthly_Balance
#  0       0x160a  ...  186.26670208571772
#  1       0x160b  ...  361.44400385378196
#  2       0x160c  ...  264.67544623342997
#  3       0x160d  ...  343.82687322383634
#  4       0x1616  ...   485.2984336755923
#  ...        ...  ...                 ...
#  49995  0x25fe5  ...  275.53956951573343
#  49996  0x25fee  ...  409.39456169535066
#  49997  0x25fef  ...   349.7263321025098
#  49998  0x25ff0  ...  463.23898098947717
#  49999  0x25ff1  ...  360.37968260123847
#  [50000 rows x 27 columns]
#---------------------------------------------------------------#
df_train.shape, df_test.shape  #  ((100000, 28), (50000, 27))
#---------------------------------------------------------------#
display(
    df_train.describe().T,
    print(),
    df_test.describe().T
)
#---------------------------------------------------------------#
display(
    df_train.describe(exclude=np.number).T,
    print(),
    df_test.describe(exclude=np.number).T
)
#---------------------------------------------------------------#
df_train['Credit_Score'].isna().sum()
(df_train.columns[:-1]!=df_test.columns).sum()

df = pd.concat([df_train, df_test], ignore_index=True)
df.shape #150000,28
#---------------------------------------------------------------#
df['Credit_Score'].isna().sum() #50000
#---------------------------------------------------------------#
df.isna().sum()
#---------------------------------------------------------------#
df.isnull().mean()*100
#---------------------------------------------------------------#
df.columns
#--------------EXAMINING DATA-----------------#
df.select_dtypes('O').info()

object_col = df.describe(include='O').columns
object_col
#---------------------------------------------------------------#
for col in object_col:
    print('Column Name: '+col)
    print("**"*20)
    print(df_train[col].value_counts(dropna=False))
    print('END', "--"*18, '\n')
#---------------------------------------------------------------#
df_copy1 = df.copy()
df_copy1.shape  #(150000, 28)
#---------------------------------------------------------------#
def text_cleaning(data):
    if pd.isna(data) or not isinstance(data, str):
        return data
    else:
        return str(data).strip('_ ,"')
#---------------------------------------------------------------#
df = df_copy1.map(text_cleaning).replace(['', 'nan', '!@9#%8', '#F%$D@*&8'], np.nan)
df
#---------------------------------------------------------------#
df.isna().sum()

#-------FIXING-------#
df.select_dtypes('O').info()
#---------------------------------------------------------------#
df['ID']                      = df.ID.apply(lambda x: int(x, 16))
df['Customer_ID']             = df.Customer_ID.apply(lambda x: int(x[4:], 16))
df['Month']                   = pd.to_datetime(df.Month, format='%B').dt.month
df['Age']                     = df.Age.astype(int)
df['SSN']                     = df.SSN.apply(lambda x: x if x is np.nan else int(str(x).replace('-', ''))).astype(float)
df['Annual_Income']           = df.Annual_Income.astype(float)
df['Num_of_Loan']             = df.Num_of_Loan.astype(int)
df['Num_of_Delayed_Payment']  = df.Num_of_Delayed_Payment.astype(float)
df['Changed_Credit_Limit']    = df.Changed_Credit_Limit.astype(float)
df['Outstanding_Debt']        = df.Outstanding_Debt.astype(float)
df['Amount_invested_monthly'] = df.Amount_invested_monthly.astype(float)
df['Monthly_Balance']         = df.Monthly_Balance.astype(float)
#--------------String to Integer----------------#
df['Occupation_Num'] = df.Occupation.astype('category').cat.codes
df['Credit_Mix_Num'] = df.Credit_Mix.astype('category').cat.codes
df['Payment_of_Min_Amount_Num'] = df.Payment_of_Min_Amount.astype('category').cat.codes
df['Payment_Behaviour_Num'] = df.Payment_Behaviour.astype('category').cat.codes
#----------------------------------------------------------------#
def Month_Converter(x):
    if pd.notnull(x):
        num1 = int(x.split(' ')[0])
        num2 = int(x.split(' ')[3])

        return (num1 * 12) + num2
    else:
        return x
# Month_Converter('3 Years and 1 Months')
df['Credit_History_Age'] = df.Credit_History_Age.apply(lambda x: Month_Converter(x)).astype(float)
df.groupby('Customer_ID')['Credit_History_Age'].apply(list)
df['Type_of_Loan'].value_counts(dropna=False).head(20)
#----------------------------------------------------------------#
df['Type_of_Loan'] = df['Type_of_Loan'].apply(lambda x: x.lower().replace('and ', '').replace(', ', ',').strip() if pd.notna(x) else x)
df.groupby('Customer_ID')['Type_of_Loan'].value_counts(dropna=False)
df.groupby('Customer_ID')['Type_of_Loan'].apply(list)
#----------------------------------------------------------------#
def get_Diff_Values_Colum(df_column, diff_value=[], sep=',', replace=''):
    column = df_column.dropna()
    for i in column:
        if sep not in i and i not in diff_value:
            diff_value.append(i)
        else:
            for data in map(lambda x:x.strip(), re.sub(replace, '', i).split(sep)):
                if not data in diff_value:
                    diff_value.append(data)
    return dict(enumerate(sorted(diff_value)))
#----------------------------------------------------------------#
get_Diff_Values_Colum(df['Type_of_Loan'])
#----------------------------------------------------------------#
# Reassign and Show Function
def Object_NaN_Values_Reassign_Group_Mode(df, groupby, column, inplace=True):
    import scipy.stats as stats
    # Assigning Wrong values Make Simple Function
    def make_NaN_and_fill_mode(df, groupby, column, inplace=False):  # parametrelerini senin koduna göre uyarla
        fill_func = lambda x: x.fillna(x.mode().iloc[0]) if not x.mode().empty else x

        if inplace:
            df[column] = df.groupby(groupby)[column].transform(fill_func)
            return df
        else:
            return df.groupby(groupby)[column].transform(fill_func)

    # Run
    if inplace:
        # Before Assigning NaN values
        if df[column].value_counts(dropna=False).index.isna().sum():
            x = df[column].value_counts(dropna=False).loc[[np.nan]]
            print(f'\nBefore Assigning: {column}:', f'have {x.values[0]} NaN Values', end='\n')

        a = df.groupby(groupby)[column].apply(list)
        print(f'\nBefore Assigning Example {column}:\n', *a.head().values, sep='\n', end='\n')

        # Assigning
        make_NaN_and_fill_mode(df, groupby, column, inplace)

        # After Assigning NaN values
        if df[column].value_counts(dropna=False).index.isna().sum():
            y = df[column].value_counts(dropna=False).loc[[np.nan]]
            print(f'\nBefore Assigning: {column}:', f'have {y.values[0]} NaN Values', end='\n')

        b = df.groupby(groupby)[column].apply(list)
        print(f'\nAfter Assigning Example {column}:\n', *b.head().values, sep='\n', end='\n')
    else:
        # Show
        return make_NaN_and_fill_mode(df, groupby, column, inplace)
#---------------------------------------------------------------------#
df_copy2 = df.copy()
df_copy2.shape #150000,32
#---------------------------------------------------------------------#
df = df_copy2
df.info()
#---------------------------------------------------------------------#
df['Name'].value_counts(dropna=False).head()
Object_NaN_Values_Reassign_Group_Mode(df, 'Customer_ID', 'Name')
df['Occupation'].value_counts(dropna=False)
Object_NaN_Values_Reassign_Group_Mode(df, 'Customer_ID', 'Occupation')
df.groupby('Customer_ID')['Type_of_Loan'].value_counts(dropna=False)
df['Type_of_Loan'] = df['Type_of_Loan'].fillna('No Data')
df['Credit_Mix'].value_counts(dropna=False)
Object_NaN_Values_Reassign_Group_Mode(df, 'Customer_ID', 'Credit_Mix')
df['Payment_of_Min_Amount'].value_counts(dropna=False)
df['Payment_Behaviour'].value_counts(dropna=False)
Object_NaN_Values_Reassign_Group_Mode(df, 'Customer_ID', 'Payment_Behaviour')
#----------------------------------------------------------------------#
df_copy3 = df.copy()
df_copy3.shape #150000,32
df = df_copy3
df.info()
#----------------------------------------------------------------------#
df['Customer_ID'].nunique() #12500
#----------------------------------------------------------------------#
# Define Outlier Range
def get_iqr_lower_upper(df, column, multiply=1.5):
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1

    lower = q1 - iqr * multiply
    upper = q3 + iqr * multiply
    affect = df.loc[(df[column] < lower) | (df[column] > upper)].shape
    print('Outliers:', affect)
    return lower, upper
#----------------------------------------------------------------------#
# Reassign Wrong Values and Show Function
def Numeric_Wrong_Values_Reassign_Group_Min_Max(df, groupby, column, inplace=True):
    # Identify Wrong values Range
    def get_group_min_max(df, groupby, column):
        cur = df[df[column].notna()].groupby(groupby)[column].apply(list)

        # DÜZELTME 1: stats.mode(x) yerine stats.mode(x).mode yaptık.
        # Böylece nesneyi değil, içindeki sayıyı alıyoruz.
        x, y = cur.apply(lambda x: stats.mode(x).mode).apply([min, max])

        # Eğer sonuç tek elemanlı bir array ise içindeki sayıyı al (Scalar'a çevir)
        if hasattr(x, 'item'): x = x.item()
        if hasattr(y, 'item'): y = y.item()

        return x, y

    # Assigning Wrong values
    def make_group_NaN_and_fill_mode(df, groupby, column, inplace=True):
        df_dropped = df[df[column].notna()].groupby(groupby)[column].apply(list)

        # DÜZELTME 2: Burada da aynı şekilde .mode eklendi
        x, y = df_dropped.apply(lambda x: stats.mode(x).mode).apply([min, max])

        # Array'den sayıya çevirme güvenliği
        if hasattr(x, 'item'): x = x.item()
        if hasattr(y, 'item'): y = y.item()

        mini, maxi = x, y

        # assign Wrong Values to NaN
        # mini ve maxi artık sayı olduğu için bu kıyaslama hatasız çalışıyor
        col = df[column].apply(lambda x: np.nan if ((x < mini) | (x > maxi)) else x)

        # fill with local mode
        # DÜZELTME 3: Pandas'ın kendi mode fonksiyonunu kullandım
        mode_by_group = df.groupby(groupby)[column].transform(
            lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
        result = col.fillna(mode_by_group)

        if inplace:
            df[column] = result
        else:
            return result

    # Run
    if inplace:
        if df[column].value_counts(dropna=False).index.isna().sum():
            x = df[column].value_counts(dropna=False).loc[[np.nan]]
            print(f'\nBefore Assigning: {column}:', f'have {x.values[0]} NaN Values', end='\n')

        print("\nExisting Min, Max Values:", df[column].apply([min, max]), sep='\n', end='\n')

        mini, maxi = get_group_min_max(df, groupby, column)
        print(f"\nGroupby by {groupby}'s Actual min, max Values:", f'min:\t{mini},\nmax:\t{maxi}', sep='\n', end='\n')

        a = df.groupby(groupby)[column].apply(list)
        print(f'\nBefore Assigning Example {column}:\n', *a.head().values, sep='\n', end='\n')

        # Assigning
        make_group_NaN_and_fill_mode(df, groupby, column, inplace)

        # After Assigning NaN values
        if df[column].value_counts(dropna=False).index.isna().sum():
            y = df[column].value_counts(dropna=False).loc[[np.nan]]
            print(f'\nBefore Assigning: {column}:', f'have {y.values[0]} NaN Values', end='\n')

        b = df.groupby(groupby)[column].apply(list)
        print(f'\nAfter Assigning Example {column}:\n', *b.head().values, sep='\n', end='\n')
    else:
        return make_group_NaN_and_fill_mode(df, groupby, column, inplace)
#---------------------------------------------------------------------------------#
df.describe().columns
df['ID'].nunique() #150000
df['Month'].value_counts()
Numeric_Wrong_Values_Reassign_Group_Min_Max(df, 'Customer_ID', 'Age')
# Check Outlier
get_iqr_lower_upper(df, 'Age') #(0,32) - (-0.5,67.5)
#---------------------------------------------------------------------------------#
df.SSN.value_counts(dropna=False)
Numeric_Wrong_Values_Reassign_Group_Min_Max(df, 'Customer_ID', 'SSN')
df.Annual_Income.value_counts(dropna=False)
Numeric_Wrong_Values_Reassign_Group_Min_Max(df, 'Customer_ID', 'Annual_Income')
df.Monthly_Inhand_Salary.value_counts(dropna=False)
Numeric_Wrong_Values_Reassign_Group_Min_Max(df, 'Customer_ID', 'Monthly_Inhand_Salary')
df.Num_Bank_Accounts.value_counts(dropna=False)
Numeric_Wrong_Values_Reassign_Group_Min_Max(df, 'Customer_ID', 'Num_Bank_Accounts')
df_train.Num_Credit_Card.value_counts(dropna=False)
Numeric_Wrong_Values_Reassign_Group_Min_Max(df, 'Customer_ID', 'Num_Credit_Card')
df.Interest_Rate.value_counts(dropna=False)
Numeric_Wrong_Values_Reassign_Group_Min_Max(df, 'Customer_ID', 'Interest_Rate')
df.Num_of_Loan.value_counts(dropna=False)
Numeric_Wrong_Values_Reassign_Group_Min_Max(df, 'Customer_ID', 'Num_of_Loan')
df.Delay_from_due_date.value_counts(dropna=False)
Numeric_Wrong_Values_Reassign_Group_Min_Max(df, 'Customer_ID', 'Delay_from_due_date')
df.Num_of_Delayed_Payment.value_counts(dropna=False)
Numeric_Wrong_Values_Reassign_Group_Min_Max(df, 'Customer_ID', 'Num_of_Delayed_Payment')
df.Changed_Credit_Limit.value_counts(dropna=False)
Numeric_Wrong_Values_Reassign_Group_Min_Max(df, 'Customer_ID', 'Changed_Credit_Limit')
df.Num_Credit_Inquiries.value_counts(dropna=False)
Numeric_Wrong_Values_Reassign_Group_Min_Max(df, 'Customer_ID', 'Num_Credit_Inquiries')
df.Outstanding_Debt.value_counts(dropna=False)
Numeric_Wrong_Values_Reassign_Group_Min_Max(df, 'Customer_ID', 'Outstanding_Debt')
df.Credit_Utilization_Ratio.value_counts(dropna=False)
df.Credit_Utilization_Ratio.isna().sum() # 0
df.Credit_History_Age.value_counts(dropna=False)
df['Credit_History_Age'] = df.groupby('Customer_ID')['Credit_History_Age'].transform(lambda x: x.interpolate().bfill().ffill())
df.Total_EMI_per_month.value_counts(dropna=False)
Numeric_Wrong_Values_Reassign_Group_Min_Max(df, 'Customer_ID', 'Total_EMI_per_month')
df.Amount_invested_monthly.value_counts(dropna=False)
Numeric_Wrong_Values_Reassign_Group_Min_Max(df, 'Customer_ID', 'Amount_invested_monthly')
df.Monthly_Balance.value_counts(dropna=False)
Numeric_Wrong_Values_Reassign_Group_Min_Max(df, 'Customer_ID', 'Monthly_Balance')
#---------------------------------------------------------------------------------------------------#
df
df.isna().sum()
df.info()
df.to_csv("clean_credit_score_classification.csv", index=False)
df = pd.read_csv('clean_credit_score_classification.csv', low_memory=False)
#----------------------------------------------------------------------------------------------------------------------#
df[df['Num_Bank_Accounts']<0]
df[df['Num_Bank_Accounts']<0]['Customer_ID'].unique()
df[df['Customer_ID']==22931]
df.loc[df['Num_Bank_Accounts']<0, 'Num_Bank_Accounts'] = 0
df[df['Delay_from_due_date']<0]
df[df['Delay_from_due_date']<0]['Customer_ID'].unique()
df[df['Customer_ID']==48234].iloc[:,0:15]
df.loc[df['Delay_from_due_date']<0, 'Delay_from_due_date'] = None
Numeric_Wrong_Values_Reassign_Group_Min_Max(df, 'Customer_ID', 'Delay_from_due_date')
df[df['Num_of_Delayed_Payment']<0]
df[df['Num_of_Delayed_Payment']<0]['Customer_ID'].unique()
df[df['Customer_ID']==8625].iloc[:,0:20]
df.loc[df['Num_of_Delayed_Payment']<0, 'Num_of_Delayed_Payment'] = None
Numeric_Wrong_Values_Reassign_Group_Min_Max(df, 'Customer_ID', 'Num_of_Delayed_Payment')
df[df['Monthly_Balance']<0]
df[df['Monthly_Balance']<0]['Customer_ID'].unique()
df[df['Customer_ID']==23184]
df.loc[df['Monthly_Balance']<0, 'Monthly_Balance'] = None
Numeric_Wrong_Values_Reassign_Group_Min_Max(df, 'Customer_ID', 'Monthly_Balance')
df[df['Amount_invested_monthly']>=10000]
df[df['Amount_invested_monthly']>=10000]['Customer_ID'].unique()
df[df['Customer_ID']==44897]
df['Amount_invested_monthly'].plot(kind='box', vert=False)
df.loc[df['Amount_invested_monthly']>=10000, 'Amount_invested_monthly'] = None
df['Amount_invested_monthly'].plot(kind='box', vert=False)
#----------------------------------------------------------------------------------------------------------------------#
# fill group Mode
df['Amount_invested_monthly'] = df.groupby('Customer_ID')['Amount_invested_monthly'].transform(lambda x: x.mode()[0] if not x.mode().empty else np.NaN)
# train check
df[df['Credit_Score'].notna()].info()
# train save
df[df['Credit_Score'].notna()].to_csv("train.csv", index=False)
# test check
df[df['Credit_Score'].isna()].info()
# test save
df[df['Credit_Score'].isna()].drop(columns='Credit_Score').to_csv("test.csv", index=False)
#----------#
#-DOWNLOAD-#
from IPython.display import FileLink, FileLinks
train_file = FileLink(r'train.csv', result_html_prefix="Click here to download: ")
test_file = FileLink(r'test.csv', result_html_prefix="Click here to download: ")

display(train_file, test_file)